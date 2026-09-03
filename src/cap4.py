import os
import glob
import re
import json
import cv2
import torch
import numpy as np
from PIL import Image
import open_clip
from huggingface_hub import hf_hub_download
from datetime import datetime

# Optional FAISS import for high-speed vector clustering with a safe fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

print("Initializing Capability 4: Advanced Unsupervised Discovery & Spatial Clustering Engine...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

print(f"Loading RemoteCLIP ({model_name}) weights offline for spatial intelligence...")
try:
    ckpt_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", local_files_only=True)
except Exception:
    ckpt_path = f"checkpoints/RemoteCLIP-{model_name}.pt"

if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
else:
    raise FileNotFoundError(f"RemoteCLIP weights not found at '{ckpt_path}'.")

model = model.to(device).eval()
print("Model loaded locally and ready!\n")

# ==========================================
# GEOSPATIAL BOUNDING BOXES (WGS 84)
# ==========================================
DATASET_BOUNDS = {
    "data_kanpur_5km": [80.3000, 26.4000, 80.3500, 26.4500],   
    "data_kharagpur_5km": [87.3000, 22.3000, 87.3500, 22.3500] 
}

def pixel_to_latlon(x, y, img_w, img_h, bounds):
    min_lon, min_lat, max_lon, max_lat = bounds
    lon = min_lon + (x / img_w) * (max_lon - min_lon)
    lat = max_lat - (y / img_h) * (max_lat - min_lat)
    return [round(float(lon), 6), round(float(lat), 6)]

class AdvancedDiscoveryAndClusteringEngine:
    """
    Enhanced Capability 4: Unsupervised Discovery & Spatial Clustering.
    Features:
      1. Multi-Band Vector Fusion (Combines True Color, NIR, and SWIR fingerprints).
      2. Geographic Proximity & Coordinate Mapping (Converts pixel coordinates to real WGS84 GeoJSON).
      3. Automatic Cluster Grouping (Groups matched trouble spots into spatial heatmaps).
    """
    def __init__(self, device=device):
        self.device = device

    def scan_multiband_records(self, base_folder="./satellite_datasets"):
        """Scans regional folders to aggregate multi-band assets (RGB, NIR, SWIR) per tile."""
        if not os.path.exists(base_folder):
            return []
        
        valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        records_map = {}

        for root, dirs, files in os.walk(base_folder):
            rel_path = os.path.relpath(root, base_folder)
            parts = rel_path.split(os.sep)
            placename = parts[0] if len(parts) > 0 and parts[0] != "." else "Unknown_Location"
            bounds = DATASET_BOUNDS.get(placename, None)
            
            for f in files:
                if f.lower().endswith(valid_extensions):
                    # Strip band suffixes to unify tiles across multi-spectral folders
                    base_name = re.sub(r'(_true_color|_false_color_nir|_agriculture_swir|_rgb|_nir|_swir)', '', f, flags=re.IGNORECASE)
                    
                    if base_name not in records_map:
                        records_map[base_name] = {
                            "base_name": base_name,
                            "filename": f,
                            "placename": placename,
                            "bounds": bounds,
                            "bands": {}
                        }
                    
                    layer_type = "true_color"
                    if "false_color" in root.lower() or "nir" in root.lower():
                        layer_type = "false_color_nir"
                    elif "swir" in root.lower() or "agriculture" in root.lower():
                        layer_type = "agriculture_swir"
                        
                    records_map[base_name]["bands"][layer_type] = os.path.join(root, f)

        # Ensure every record prioritizes True Color for primary visual embedding
        valid_records = []
        for base_name, data in records_map.items():
            tc_path = data["bands"].get("true_color") or next(iter(data["bands"].values()), None)
            if tc_path and os.path.exists(tc_path):
                data["primary_path"] = tc_path
                valid_records.append(data)

        valid_records.sort(key=lambda x: x["filename"])
        return valid_records

    @torch.no_grad()
    def get_multiband_embeddings(self, records, batch_size=32):
        """
        Computes robust multi-spectral vector embeddings by averaging 
        available bands (RGB, NIR, SWIR) into a unified visual signature.
        """
        if not records:
            return np.array([])
        
        all_features = []
        for i in range(0, len(records), batch_size):
            batch_recs = records[i:i + batch_size]
            batch_tensors = []
            
            for rec in batch_recs:
                band_embeddings = []
                for band_name, p in rec["bands"].items():
                    try:
                        img = preprocess(Image.open(p).convert("RGB"))
                        band_embeddings.append(img)
                    except Exception:
                        continue
                
                if band_embeddings:
                    # Stack and average across available spectral bands to form an omni-spectral signature
                    stack = torch.stack(band_embeddings).to(self.device)
                    with torch.autocast(device_type=self.device.type):
                        feat = model.encode_image(stack)
                    feat /= feat.norm(dim=-1, keepdim=True)
                    mean_feat = feat.mean(dim=0)
                    mean_feat /= mean_feat.norm(dim=-1, keepdim=True)
                    batch_tensors.append(mean_feat)
                else:
                    # Fallback to zero vector if unreadable
                    batch_tensors.append(torch.zeros(768, device=self.device))
            
            if batch_tensors:
                stacked_batch = torch.stack(batch_tensors)
                all_features.append(stacked_batch.cpu().to(torch.float32).numpy())
                
        return np.vstack(all_features) if all_features else np.array([])

    def discover_and_cluster_spots(self, reference_identifier, base_folder="./satellite_datasets", top_k=10, similarity_threshold=65.0):
        """
        Advanced Unsupervised Discovery:
        Finds a reference trouble spot, computes multi-spectral embeddings, clusters 
        similar geographic spots across the map, and maps them to WGS84 GeoJSON.
        """
        records = self.scan_multiband_records(base_folder)
        if not records:
            print("No valid multi-band image records found.")
            return [], None

        # Resolve reference tile precisely
        ref_rec = None
        for r in records:
            if r["filename"] == os.path.basename(reference_identifier) or r["base_name"] == reference_identifier or os.path.abspath(r["primary_path"]) == os.path.abspath(reference_identifier):
                ref_rec = r
                break

        if not ref_rec:
            print(f"Error: Reference trouble spot '{reference_identifier}' could not be located.")
            return [], None

        print(f"\n[Spatial Discovery] Computing multi-spectral feature matrix across regional assets...")
        embeddings = self.get_multiband_embeddings(records)
        
        ref_idx = records.index(ref_rec)
        ref_emb = embeddings[ref_idx:ref_idx+1]

        if embeddings.size == 0 or ref_emb.size == 0:
            print("Error: Multi-band embedding generation failed.")
            return [], None

        # FAISS Acceleration with NumPy fallback
        if FAISS_AVAILABLE and embeddings.shape[0] > 5:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings.astype(np.float32))
            
            k = min(len(records), top_k + 1)
            distances, ranked_indices = index.search(ref_emb.astype(np.float32), k)
            similarities = distances[0] * 100
            ranked_indices = ranked_indices[0]
        else:
            similarities = (embeddings @ ref_emb.T).squeeze(-1) * 100
            ranked_indices = np.argsort(similarities)[::-1]

        clustered_spots = []
        geojson_features = []

        for idx in ranked_indices:
            if idx == -1: 
                continue
            score = float(similarities[list(ranked_indices).index(idx)] if FAISS_AVAILABLE and embeddings.shape[0] > 5 else similarities[idx])
            
            rec = records[idx]
            if rec["base_name"] == ref_rec["base_name"]:
                continue
                
            if score < similarity_threshold:
                break
                
            spot_info = {
                "placename": rec["placename"],
                "filename": rec["filename"],
                "primary_path": rec["primary_path"],
                "match_score_pct": float(round(score, 2)),
                "bounds": rec["bounds"]
            }
            clustered_spots.append(spot_info)

            # Build GIS GeoJSON Polygon Feature for mapping
            bounds = rec["bounds"]
            if bounds:
                min_lon, min_lat, max_lon, max_lat = bounds
                poly_coords = [[[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat], [min_lon, min_lat]]]
                geometry = {"type": "Polygon", "coordinates": poly_coords}
            else:
                geometry = None

            geojson_features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "cluster_group": f"Similar_to_{ref_rec['base_name']}",
                    "placename": rec["placename"],
                    "filename": rec["filename"],
                    "match_confidence": spot_info["match_score_pct"]
                }
            })

            if len(clustered_spots) >= top_k:
                break

        # Construct GeoJSON FeatureCollection
        feature_collection = {
            "type": "FeatureCollection",
            "name": f"Discovered_Clusters_{ref_rec['base_name']}",
            "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
            "features": geojson_features
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        geojson_path = f"discovered_clusters_{ref_rec['placename']}_{timestamp}.geojson"
        with open(geojson_path, 'w') as f:
            json.dump(feature_collection, f, indent=4)

        return clustered_spots, geojson_path

def interactive_advanced_discovery_menu():
    base_dir = "./satellite_datasets"
    if not os.path.exists(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return

    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    if not folders:
        print(f"No datasets found in '{base_dir}'.")
        return

    print("=" * 60)
    print(" CAPABILITY 4: ADVANCED SPATIAL DISCOVERY & CLUSTERING")
    print("=" * 60)
    for i, folder in enumerate(folders, 1):
        print(f"   [{i}] {folder}")
    
    try:
        folder_choice = int(input("\nSelect dataset region folder to run cluster analysis: ").strip()) - 1
        if not (0 <= folder_choice < len(folders)):
            print("Invalid selection.")
            return
    except ValueError:
        print("Invalid input.")
        return
        
    target_directory = os.path.join(base_dir, folders[folder_choice])
    
    engine = AdvancedDiscoveryAndClusteringEngine()
    records = engine.scan_multiband_records(target_directory)
    
    if not records:
        print("No valid imagery records found in this dataset.")
        return

    print(f"\nAvailable reference tiles in {folders[folder_choice]}:")
    for idx, r in enumerate(records[:12], 1):
        print(f"   [{idx}] {r['filename']} (Bands found: {list(r['bands'].keys())})")
    if len(records) > 12:
        print(f"   ... and {len(records) - 12} more records.")

    ref_input = input("\nEnter reference trouble spot filename or identifier: ").strip().strip('"').strip("'")
    if not ref_input:
        print("No reference provided.")
        return

    matches, geojson_path = engine.discover_and_cluster_spots(ref_input, base_folder=target_directory, top_k=5, similarity_threshold=55.0)

    print("\n" + "=" * 60)
    print(f" CLUSTERED TROUBLE SPOTS SURFACED & MAPPED ")
    print("=" * 60)
    if matches:
        for idx, match in enumerate(matches, 1):
            print(f"  {idx}. [Confidence: {match['match_score_pct']}%] | Region: {match['placename']} | Tile: {match['filename']}")
        print(f"\n[+] GIS-ready Cluster GeoJSON exported successfully to: '{geojson_path}'")
        print("    (You can drag and drop this file directly into QGIS or Mapbox!)")
    else:
        print("  ❌ No similar trouble spots found exceeding the spatial similarity threshold.")
    print("=" * 60)

if __name__ == "__main__":
    interactive_advanced_discovery_menu()