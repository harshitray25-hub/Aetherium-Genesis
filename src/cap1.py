import os
import re
import glob
import json
import torch
import gc
import numpy as np
from PIL import Image
import open_clip
from huggingface_hub import hf_hub_download
from datetime import datetime
import concurrent.futures

# Optional FAISS import for sub-millisecond vector indexing with a safe fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# ==========================================
# GEOSPATIAL BOUNDING BOXES (WGS 84)
# ==========================================
DATASET_BOUNDS = {
    "data_kanpur_5km": [80.3000, 26.4000, 80.3500, 26.4500],   
    "data_kharagpur_5km": [87.3000, 22.3000, 87.3500, 22.3500] 
}

print("Initializing Fully Local OpenCLIP RemoteCLIP Engine...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

ckpt_path = "checkpoints/RemoteCLIP-ViT-L-14.pt"
if not os.path.exists(ckpt_path):
    try:
        print("Local weights not found, attempting Hugging Face Hub download...")
        ckpt_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", local_files_only=False)
    except Exception as e:
        print(f"Warning: Download failed ({e}).")

if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
else:
    raise FileNotFoundError(f"RemoteCLIP weights not found at '{ckpt_path}'.")

model = model.to(device).eval()
print(f"Model loaded locally on [{device.type.upper()}] and ready!\n")

ARTIFACT_TEXT_FEATURES = None

def get_artifact_text_features():
    global ARTIFACT_TEXT_FEATURES
    if ARTIFACT_TEXT_FEATURES is None:
        queries = [
            "clear satellite view of ground features",
            "clouds, thick fog, smoke haze, smog, or dust storm blocking the ground view",
            "long shadows, low sun angle geometry, off-nadir distortion"
        ]
        text = tokenizer(queries).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type):
            features = model.encode_text(text)
            features /= features.norm(dim=-1, keepdim=True)
            ARTIFACT_TEXT_FEATURES = features.cpu().to(torch.float32).numpy()
    return ARTIFACT_TEXT_FEATURES

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def parse_date_from_filename(filename):
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    return datetime.strptime(match.group(0), "%Y-%m-%d") if match else None

def check_bbox_intersection(tile_bounds, search_aoi):
    if not tile_bounds or not search_aoi:
        return True
    t_min_lon, t_min_lat, t_max_lon, t_max_lat = tile_bounds
    s_min_lon, s_min_lat, s_max_lon, s_max_lat = search_aoi
    
    if t_max_lon < s_min_lon or t_min_lon > s_max_lon:
        return False
    if t_max_lat < s_min_lat or t_min_lat > s_max_lat:
        return False
    return True

def scan_dataset_folder(base_folder="./satellite_datasets", layer_type="True_Color"):
    valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    image_records = []

    keywords_map = {
        "True_Color": ["true_color", "true-color", "rgb", "truecolor"],
        "False_Color": ["false_color", "false-color", "nir", "falsecolor"],
        "Agriculture_SWIR": ["swir", "agriculture", "agriculture_swir"]
    }
    target_keywords = keywords_map.get(layer_type, ["true_color"])

    for root, dirs, files in os.walk(base_folder):
        is_target_layer = any(term in root.lower() for term in target_keywords)
        rel_path = os.path.relpath(root, base_folder)
        parts = rel_path.split(os.sep)
        placename = parts[0] if len(parts) > 0 and parts[0] != "." else "Unknown_Location"
        
        bounds = DATASET_BOUNDS.get(placename, None)
        
        for f in files:
            if f.lower().endswith(valid_extensions) and (is_target_layer or layer_type == "Fallback"):
                image_records.append({
                    "path": os.path.join(root, f),
                    "filename": f,
                    "placename": placename,
                    "bounds": bounds,
                    "date": parse_date_from_filename(f),
                    "layer_source": layer_type
                })

    if not image_records and layer_type != "Fallback":
        return scan_dataset_folder(base_folder, layer_type="Fallback")

    image_records.sort(key=lambda x: x["filename"])
    return image_records

def load_and_preprocess(path):
    try:
        return preprocess(Image.open(path).convert("RGB"))
    except Exception:
        return None

@torch.no_grad()
def get_image_embeddings(image_paths, batch_size=32):
    if not image_paths:
        return np.array([])
    
    all_features = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images_list = list(executor.map(load_and_preprocess, batch_paths))
            valid_images = [img for img in images_list if img is not None]
            
            if not valid_images: continue
            
            images = torch.stack(valid_images).to(device)
            with torch.autocast(device_type=device.type):
                features = model.encode_image(images)
                
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().to(torch.float32).numpy())
            
            del images, features
            gc.collect()
            if device.type == 'cuda': torch.cuda.empty_cache()
            
    return np.vstack(all_features)

@torch.no_grad()
def get_ensemble_text_embedding(query_text, enhancements):
    base_context = "high-resolution nadir satellite imagery, aerial view"
    prompts = [
        f"{query_text}, {enhancements}, {base_context}",
        f"satellite view of {query_text}",
        f"aerial photograph showing {query_text}"
    ]
    
    text = tokenizer(prompts).to(device)
    with torch.autocast(device_type=device.type):
        features = model.encode_text(text)
    
    features /= features.norm(dim=-1, keepdim=True)
    mean_feature = features.mean(dim=0, keepdim=True)
    mean_feature /= mean_feature.norm(dim=-1, keepdim=True)
    
    return mean_feature.cpu().to(torch.float32).numpy()

def load_or_compute_embeddings(records, layer_type):
    cache_path = f"embedding_cache_{layer_type}.npz"
    current_paths = [r["path"] for r in records]
    
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            path_to_emb = {p: emb for p, emb in zip(list(data["paths"]), data["embeddings"])}
            missing_paths = [p for p in current_paths if p not in path_to_emb]
            
            if missing_paths:
                print(f"Computing embeddings for {len(missing_paths)} new '{layer_type}' tiles...")
                new_embs = get_image_embeddings(missing_paths)
                for p, emb in zip(missing_paths, new_embs):
                    path_to_emb[p] = emb
                    
            final_embeddings = np.array([path_to_emb[p] for p in current_paths])
            np.savez(cache_path, paths=current_paths, embeddings=final_embeddings)
            return final_embeddings
        except Exception as e:
            print(f"Warning: Cache '{cache_path}' corrupted. Rebuilding...")
            os.remove(cache_path)

    print(f"Computing initial embeddings for all {len(current_paths)} '{layer_type}' tiles...")
    embeddings = get_image_embeddings(current_paths)
    np.savez(cache_path, paths=current_paths, embeddings=embeddings)
    return embeddings

def intelligent_layer_router(query):
    query_lower = query.lower()
    target_layers = set()
    enhancements = []
    
    if re.search(r'\b(construction|building|urban|residential|newly built|concrete|scaffold|structure|city)\b', query_lower):
        target_layers.update(["True_Color", "False_Color"])
        enhancements.append("high albedo concrete, metal rooftops, scaffolding footprints, urban infrastructure")
        
    if re.search(r'\b(road|highway|street|paved|traffic|linear|infrastructure|bridge)\b', query_lower):
        target_layers.update(["False_Color", "True_Color"])
        enhancements.append("smooth thin linear grey or white asphalt trace, transportation network")
        
    if re.search(r'\b(cleared|deforest|land clearance|site prep|logging|bare soil|earth|excavation)\b', query_lower):
        target_layers.add("False_Color")
        enhancements.append("exposed earth, deforestation, dull browns and greys replacing vegetation")
        
    if re.search(r'\b(flood|water|river|lake|drying|inundation|reservoir|pond|coast)\b', query_lower):
        target_layers.update(["Agriculture_SWIR", "False_Color"])
        enhancements.append("pitch black pixel clusters, water completely absorbing infrared light, aquatic boundaries")
        
    if re.search(r'\b(agriculture|farm|crop|field|harvest|vegetation|forest|greenery)\b', query_lower):
        target_layers.update(["False_Color", "Agriculture_SWIR"])
        enhancements.append("healthy vegetation glowing bright red, agricultural plots, organized farmland")

    if not target_layers:
        target_layers.update(["True_Color", "False_Color", "Agriculture_SWIR"])
        enhancements.append("varied terrain, landscape overview")

    combined_enhancements = ", ".join(enhancements)
    return list(target_layers), combined_enhancements

def execute_search(query_emb, records, embeddings, score_threshold=0.0, start_date=None, end_date=None, aoi=None, filter_artifacts=True):
    """
    Executes search with strict pre-filtering for spatial and temporal metadata 
    before running vector math, preventing valid candidates from being clipped out.
    """
    # Pre-filter indices based on metadata constraints
    valid_indices = []
    for idx, rec in enumerate(records):
        if start_date and rec["date"] and rec["date"] < datetime.strptime(start_date, "%Y-%m-%d"): continue
        if end_date and rec["date"] and rec["date"] > datetime.strptime(end_date, "%Y-%m-%d"): continue
        if aoi and not check_bbox_intersection(rec["bounds"], aoi): continue
        valid_indices.append(idx)
        
    if not valid_indices:
        return []
        
    filtered_records = [records[i] for i in valid_indices]
    filtered_embeddings = embeddings[valid_indices]
    
    if FAISS_AVAILABLE and filtered_embeddings.shape[0] > 5:
        dimension = filtered_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(filtered_embeddings.astype(np.float32))
        
        k = min(len(filtered_records), 100)
        distances, ranked_indices = index.search(query_emb.astype(np.float32), k)
        similarities = distances[0] * 100
        ranked_indices = ranked_indices[0]
    else:
        similarities = (filtered_embeddings @ query_emb.T).squeeze(-1) * 100
        ranked_indices = np.argsort(similarities)[::-1]
    
    artifact_text_features = get_artifact_text_features()
    logit_scale = model.logit_scale.exp().item()
    
    results = []
    seen_paths = set()
    
    for idx in ranked_indices:
        if idx == -1: continue
        score = float(similarities[list(ranked_indices).index(idx)] if FAISS_AVAILABLE and filtered_embeddings.shape[0] > 5 else similarities[idx])
        
        if score < score_threshold: break
            
        rec = filtered_records[idx]
        if rec["path"] in seen_paths: continue
        seen_paths.add(rec["path"])
        
        img_emb = filtered_embeddings[idx]
        
        if filter_artifacts:
            logits = (img_emb @ artifact_text_features.T) * logit_scale
            probs = softmax(logits)
            if probs[1] * 100 > 35.0 or probs[2] * 100 > 35.0: continue
            
        results.append({
            "placename": rec["placename"],
            "filename": rec["filename"],
            "path": rec["path"],
            "layer": rec["layer_source"],
            "score": round(score, 2),
            "date": rec["date"].strftime("%Y-%m-%d") if rec["date"] else None,
            "bounds": rec["bounds"]
        })
        
        if len(results) >= 15: break
    return results

def save_search_results_geojson(results, query_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(x for x in query_name if x.isalnum() or x in " _-").strip()[:30]
    output_path = f"search_results_{safe_name}_{timestamp}.geojson"
    
    features = []
    for res in results:
        bounds = res["bounds"]
        if bounds:
            min_lon, min_lat, max_lon, max_lat = bounds
            polygon_coords = [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat]
            ]]
            geometry = {"type": "Polygon", "coordinates": polygon_coords}
        else:
            geometry = None
            
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "placename": res["placename"],
                "filename": res["filename"],
                "band_layer": res["layer"],
                "score": res["score"],
                "date": res["date"]
            }
        })
        
    feature_collection = {
        "type": "FeatureCollection",
        "name": f"GEOINT_Search_{safe_name}",
        "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
        "features": features
    }
    
    with open(output_path, 'w') as f:
        json.dump(feature_collection, f, indent=4)
    return output_path

if __name__ == "__main__":
    target_folder = "./satellite_datasets"
    os.makedirs(target_folder, exist_ok=True)

    try:
        while True:
            print("\n" + "=" * 60)
            print(f" ENTERPRISE MULTI-BAND GEOINT SEARCH (Target: '{target_folder}')")
            print("=" * 60)
            print("   1. Semantic Text Search (Pre-filtered Spatial/Temporal & GeoJSON)")
            print("   2. Image-to-Image Cross-Spectral Search")
            print("   3. Exit")
            print("=" * 60)
            
            choice = input("Enter choice (1, 2, or 3): ").strip()
            
            if choice == '3': break
                
            elif choice == '1':
                user_query = input("Enter intelligence requirement (e.g., 'newly built concrete structures near a river'): ").strip()
                if not user_query: continue
                    
                target_layers, enhancements = intelligent_layer_router(user_query)
                print(f"\n[Intelligent Router] Routing search across Spectral Bands: {', '.join(target_layers)}")
                
                use_dates = input("Apply date range filter? (y/n): ").strip().lower()
                start_date, end_date = None, None
                if use_dates == 'y':
                    start_date = input("  Enter start date (YYYY-MM-DD or blank): ").strip() or None
                    end_date = input("  Enter end date (YYYY-MM-DD or blank): ").strip() or None

                use_aoi = input("Filter by Area of Interest (AOI)? (y/n): ").strip().lower()
                aoi_bounds = None
                if use_aoi == 'y':
                    try:
                        aoi_str = input("Enter bounds (min_lon, min_lat, max_lon, max_lat): ")
                        aoi_bounds = [float(x.strip()) for x in aoi_str.split(',')]
                    except:
                        print("Invalid bounds format. Proceeding without spatial filter.")

                all_records = []
                all_embeddings = []
                
                for layer in target_layers:
                    records = scan_dataset_folder(target_folder, layer)
                    if not records: continue
                    embs = load_or_compute_embeddings(records, layer)
                    all_records.extend(records)
                    all_embeddings.append(embs)
                
                if not all_records:
                    print("  ❌ No tiles found across any of the required spectral bands.")
                    continue
                    
                combined_embeddings = np.vstack(all_embeddings)
                text_emb = get_ensemble_text_embedding(user_query, enhancements)
                
                results = execute_search(text_emb, all_records, combined_embeddings, start_date=start_date, end_date=end_date, aoi=aoi_bounds)
                
                print(f"\n--- RANKED GEOINT RESULTS FOR: '{user_query}' ---")
                if results:
                    for idx, res in enumerate(results, 1):
                        print(f"  {idx}. [Score: {res['score']}%] | Band: {res['layer']} | Date: {res['date']} | Tile: {res['filename']}")
                    out_geojson = save_search_results_geojson(results, user_query)
                    print(f"\nGIS-ready GeoJSON exported for mapping: {out_geojson}")
                else:
                    print("  ❌ No tiles matched the required spatial, temporal, and semantic thresholds.")
                    
            elif choice == '2':
                img_path = input("Enter filename or relative path of reference tile: ").strip().strip('"').strip("'")
                if not img_path: continue
                
                found_ref = None
                ref_layer = "True_Color"
                for layer in ["True_Color", "False_Color", "Agriculture_SWIR"]:
                    recs = scan_dataset_folder(target_folder, layer)
                    match = next((r for r in recs if r["filename"] == os.path.basename(img_path) or os.path.abspath(r["path"]) == os.path.abspath(img_path)), None)
                    if match:
                        found_ref = match["path"]
                        ref_layer = layer
                        break
                
                if not found_ref or not os.path.exists(found_ref):
                    print("  ❌ Error: Reference image tile could not be found locally across dataset folders.")
                    continue
                    
                print(f"\n[Cross-Spectral Router] Found reference in layer band: '{ref_layer}'")
                records = scan_dataset_folder(target_folder, ref_layer)
                embeddings = load_or_compute_embeddings(records, ref_layer)
                ref_emb = get_image_embeddings([found_ref])
                
                results = execute_search(ref_emb, records, embeddings)
                results = [r for r in results if os.path.abspath(r["path"]) != os.path.abspath(found_ref)]
                
                print(f"\n--- RANKED CROSS-SPECTRAL MATCHES ---")
                if results:
                    for idx, res in enumerate(results[:5], 1):
                        print(f"  {idx}. [Score: {res['score']}%] | Band: {res['layer']} | Date: {res['date']} | Tile: {res['filename']}")
                else:
                    print("  ❌ No matching cross-spectral neighbors found.")
                    
    except KeyboardInterrupt:
        print("\n\nSession terminated by operator. Safely unmounting models...")