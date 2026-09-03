import os
import glob
import re
import json
import cv2
import torch
import numpy as np
from PIL import Image, ImageEnhance
import open_clip
from huggingface_hub import hf_hub_download
from datetime import datetime
import time
import gc
import concurrent.futures
from skimage.metrics import structural_similarity as ssim

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

print("Initializing Enterprise GEOINT Engine (Capabilities 1, 2, 3 & 4)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

print(f"Loading RemoteCLIP ({model_name}) weights offline...")
try:
    ckpt_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", local_files_only=True)
except Exception:
    ckpt_path = f"checkpoints/RemoteCLIP-{model_name}.pt"

if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
else:
    print(f"Warning: Checkpoint file not found at {ckpt_path}. Ensure weights are downloaded locally.")

model = model.to(device).eval()
print(f"Model loaded locally on [{device.type.upper()}] and ready!\n")


# ==========================================
# ADVANCED TUNING & GLOBALS
# ==========================================
NDWI_WATER_THRESH = 0.10        
NDVI_VEG_THRESH = 0.25          
NDVI_DROP_MIN = 0.12            
MIN_CHANGE_AREA_PX = 400        
SSIM_THRESH = 0.55              
GLOBAL_SEASONAL_LIMIT = 0.09    
EDGE_BUFFER = 15                

MIN_ROAD_ASPECT_RATIO = 3.0     
MAX_ROAD_COMPACTNESS = 0.20     
FARM_SOLIDITY_THRESH = 0.50     
MIN_CONSTRUCTION_EXTENT = 0.25  

DATASET_BOUNDS = {
    "data_kanpur_5km": [80.3000, 26.4000, 80.3500, 26.4500],   
    "data_kharagpur_5km": [87.3000, 22.3000, 87.3500, 22.3500] 
}

ARTIFACT_TEXT_FEATURES = None

def pixel_to_latlon(x, y, img_w, img_h, bounds):
    min_lon, min_lat, max_lon, max_lat = bounds
    lon = min_lon + (x / img_w) * (max_lon - min_lon)
    lat = max_lat - (y / img_h) * (max_lat - min_lat)
    return [round(float(lon), 6), round(float(lat), 6)]

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


# ==========================================
# SPECTRAL INDEX CALCULATIONS (FIXED)
# ==========================================
def calculate_ndvi(false_color_img):
    """Calculates Normalized Difference Vegetation Index (NDVI)."""
    img_float = false_color_img.astype(np.float32)
    nir = img_float[:, :, 2]
    red = img_float[:, :, 1]
    denominator = nir + red
    denominator[denominator == 0] = 1e-5
    return (nir - red) / denominator

def calculate_ndwi(true_color_img, false_color_img):
    """Calculates Normalized Difference Water Index (NDWI)."""
    green = true_color_img.astype(np.float32)[:, :, 1]
    nir = false_color_img.astype(np.float32)[:, :, 2]
    denominator = green + nir
    denominator[denominator == 0] = 1e-5
    return (green - nir) / denominator


# ==========================================
# CAPABILITY 3: QUALITY GATING & CLEANING
# ==========================================
class MultiBandQualityEnhancementPipeline:
    def __init__(self, device=device):
        self.device = device

    def check_cloud_rejection(self, img_bgr, max_cloud_pct=15.0):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        bright = v > 180
        low_sat = s < 45
        cloud_pixels = np.sum(bright & low_sat)
        total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
        pct = (cloud_pixels / total_pixels) * 100
        return pct > max_cloud_pct, pct

    def check_black_borders_rejection(self, img_bgr, border_thickness_pct=0.05, max_edge_black_pct=25.0):
        h, w = img_bgr.shape[:2]
        tb = int(h * border_thickness_pct)
        lr = int(w * border_thickness_pct)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        top_strip = gray[:tb, :]
        bottom_strip = gray[h-tb:, :]
        left_strip = gray[:, :lr]
        right_strip = gray[:, w-lr:]

        borders = np.concatenate([top_strip.flatten(), bottom_strip.flatten(), left_strip.flatten(), right_strip.flatten()])
        black_border_pixels = np.sum(borders < 15)
        border_pct = (black_border_pixels / borders.size) * 100
        return border_pct > max_edge_black_pct, border_pct

    def validate_frame(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return True, "Error: Failed to read image file."
        is_cloudy, cloud_pct = self.check_cloud_rejection(img)
        if is_cloudy:
            return True, f"Rejected: Excessive cloud cover ({cloud_pct:.1f}% > 15%)."
        has_black_edges, edge_black_pct = self.check_black_borders_rejection(img)
        if has_black_edges:
            return True, f"Rejected: Excessive black borders ({edge_black_pct:.1f}% > 25%)."
        return False, "Passed quality gate."

    def dark_channel_dehaze(self, img_bgr, omega=0.82, patch_size=15):
        img_float = img_bgr.astype(np.float64) / 255.0
        min_channel = np.min(img_float, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark_channel = cv2.erode(min_channel, kernel)
        flat_dark = dark_channel.flatten()
        flat_img = img_float.reshape(-1, 3)
        num_search = max(int(flat_dark.size * 0.001), 1)
        indices = np.argpartition(flat_dark, -num_search)[-num_search:]
        atmospheric_light = np.mean(flat_img[indices], axis=0)
        norm_img = img_float / (atmospheric_light + 1e-6)
        norm_min = np.min(norm_img, axis=2)
        transmission = 1.0 - omega * cv2.erode(norm_min, kernel)
        transmission = np.maximum(transmission, 0.1)
        dehazed = np.empty_like(img_float)
        for i in range(3):
            dehazed[:, :, i] = (img_float[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
        dehazed = np.clip(dehazed, 0, 1)
        return (dehazed * 255).astype(np.uint8)

    def match_histogram(self, source_img, template_img):
        old_shape = source_img.shape
        source_flat = source_img.reshape(-1, 3).astype(np.float32)
        template_flat = template_img.reshape(-1, 3).astype(np.float32)
        matched = np.zeros_like(source_flat)
        for i in range(3):
            s_values, s_idx, s_counts = np.unique(source_flat[:, i], return_inverse=True, return_counts=True)
            t_values, t_counts = np.unique(template_flat[:, i], return_counts=True)
            s_quantiles = np.cumsum(s_counts).astype(np.float64) / s_counts.sum()
            t_quantiles = np.cumsum(t_counts).astype(np.float64) / t_counts.sum()
            interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
            matched[:, i] = interp_t_values[s_idx]
        return matched.reshape(old_shape).astype(np.uint8)

    def evaluate_image_clarity(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None: return 0.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharpness_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
            text_query = tokenizer(["a clear satellite view of ground features"]).to(self.device)
            with torch.no_grad():
                img_feat = model.encode_image(image)
                txt_feat = model.encode_text(text_query)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                similarity = (img_feat @ txt_feat.T).item() * 100
            return float(sharpness_score + (similarity * 5.0))
        except Exception:
            return 0.0

    def process_and_standardize_image(self, image_path, master_template):
        img = cv2.imread(image_path)
        if img is None: return None
        dehazed = self.dark_channel_dehaze(img)
        pil_img = Image.fromarray(cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB))
        pil_img = ImageEnhance.Color(pil_img).enhance(1.15)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.10)
        enhanced_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        normalized = self.match_histogram(enhanced_bgr, master_template)
        return cv2.bilateralFilter(normalized, d=5, sigmaColor=30, sigmaSpace=30)

def run_quality_pipeline(region_name, base_dir="./satellite_datasets"):
    target_directory = os.path.join(base_dir, region_name)
    master_cleaned_root = "./satellite_datasets_cleaned"
    region_output_root = os.path.join(master_cleaned_root, region_name)
    os.makedirs(region_output_root, exist_ok=True)

    target_bands = ["true_color", "false_color_nir", "agriculture_swir"]
    engine = MultiBandQualityEnhancementPipeline()

    for band_name in target_bands:
        band_dir = os.path.join(target_directory, band_name)
        if not os.path.exists(band_dir): continue

        image_files = sorted(glob.glob(os.path.join(band_dir, "*.jpg")) + glob.glob(os.path.join(band_dir, "*.png")))
        if not image_files: continue

        print(f"\n--- Processing Band: [{band_name.upper()}] ---")
        output_processed_dir = os.path.join(region_output_root, band_name)
        os.makedirs(output_processed_dir, exist_ok=True)

        valid_images = []
        for path in image_files:
            filename = os.path.basename(path)
            rejected, reason = engine.validate_frame(path)
            if rejected:
                print(f"  [❌ REJECTED] {filename} -> {reason}")
            else:
                print(f"  [✓ PASSED]   {filename}")
                valid_images.append(path)

        if not valid_images: continue

        print("Evaluating surviving images to find the master reference standard...")
        best_image_path = max(valid_images, key=lambda p: engine.evaluate_image_clarity(p))
        master_template = cv2.imread(best_image_path)
        print(f"[*] Master Reference Standard: {os.path.basename(best_image_path)}")

        for path in valid_images:
            filename = os.path.basename(path)
            cleaned_img = engine.process_and_standardize_image(path, master_template)
            if cleaned_img is not None:
                cv2.imwrite(os.path.join(output_processed_dir, filename), cleaned_img)
    
    print(f"\n[+] Quality & Standardization complete. Cleaned data saved to: '{region_output_root}'")
    return region_output_root


# ==========================================
# CAPABILITY 2: CHANGE DETECTION
# ==========================================
def load_layer_cd(date_prefix, layer_name, folder_path):
    files = glob.glob(os.path.join(folder_path, f"{date_prefix}*.jpg")) + glob.glob(os.path.join(folder_path, f"{date_prefix}*.png"))
    return cv2.imread(files[0]) if files else None

def align_shapes(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        _, warp_matrix = cv2.findTransformECC(g1, g2, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        aligned_img2 = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return img1, aligned_img2
    except Exception:
        return img1, img2

def extract_change_clusters(mask, category_name, color, events_list, img_shape, check_linear=False, skip=False, strict_mode=False):
    if skip:
        events_list.append({"type": f"Global Shift ({category_name})", "center": (0, 0), "contour": None, "area": 0, "color": (128, 128, 128), "is_global": True})
        return events_list

    active_min_area = MIN_CHANGE_AREA_PX * 2 if strict_mode else MIN_CHANGE_AREA_PX
    active_min_extent = 0.40 if strict_mode else MIN_CONSTRUCTION_EXTENT
    k_size = 9 if "Clearance" in category_name or "Growth" in category_name else 15 

    merge_kernel = np.ones((k_size, k_size), np.uint8) 
    merged_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, merge_kernel)
    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = img_shape[:2]

    for c in contours:
        area = cv2.contourArea(c)
        if area > active_min_area:
            x_int, y_int, w, h = cv2.boundingRect(c)
            if x_int <= EDGE_BUFFER or y_int <= EDGE_BUFFER or (x_int + w) >= (img_w - EDGE_BUFFER) or (y_int + h) >= (img_h - EDGE_BUFFER):
                continue

            rect = cv2.minAreaRect(c)
            w_rect, h_rect = rect[1]
            aspect_ratio = max(w_rect, h_rect) / (min(w_rect, h_rect) + 1e-5)
            
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            extent = float(area) / (w * h) if w * h > 0 else 0
            
            perimeter = cv2.arcLength(c, True)
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
                
            final_category, final_color = category_name, color

            if "Clearance" in category_name and solidity > FARM_SOLIDITY_THRESH and area > 1000:
                final_category, final_color = "Agricultural Harvest", (144, 238, 144) 
            
            if check_linear and final_category != "Agricultural Harvest":
                if extent < active_min_extent: continue
                kernel = np.ones((3,3), np.uint8)
                edges = cv2.Canny(cv2.dilate(merged_mask[y_int:y_int+h, x_int:x_int+w], kernel, iterations=1), 50, 150, apertureSize=3)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=25, maxLineGap=10)
                if lines is not None and len(lines) > 0:
                    if aspect_ratio >= MIN_ROAD_ASPECT_RATIO and compactness <= MAX_ROAD_COMPACTNESS:
                        final_category, final_color = "Road Development", (0, 165, 255) 
                    else:
                        final_category, final_color = "Construction", (255, 0, 255) 
                    
            approx_contour = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            events_list.append({"type": final_category, "center": (cx, cy), "contour": approx_contour, "area": int(area), "color": final_color, "is_global": False})
    return events_list

def generate_dashboard_image(t2_rgb, events, d1, d2):
    h, w = t2_rgb.shape[:2]
    map_img = t2_rgb.copy()
    overlay = t2_rgb.copy()
    for e in events:
        if not e["is_global"] and e["contour"] is not None:
            cv2.drawContours(overlay, [e["contour"]], -1, e["color"], -1)
    cv2.addWeighted(overlay, 0.4, map_img, 0.6, 0, map_img)
    
    for e in events:
        if not e["is_global"] and e["contour"] is not None:
            cv2.drawContours(map_img, [e["contour"]], -1, e["color"], 2)
            cv2.circle(map_img, e["center"], 3, (255, 255, 255), -1)

    grouped_events = {}
    for e in events:
        grouped_events.setdefault(e["type"], {"color": e["color"], "count": 0, "area": 0, "is_global": e["is_global"]})
        grouped_events[e["type"]]["count"] += 1
        grouped_events[e["type"]]["area"] += e["area"]

    panel_height = 90 + (len(grouped_events) * 35) if grouped_events else 125
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    
    cv2.putText(panel, f"ENTERPRISE TIMELINE: {d1} to {d2}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(panel, (15, 40), (w - 15, 40), (100, 100, 100), 1)
    
    y_offset = 70
    if not grouped_events:
        cv2.putText(panel, "No significant structural changes detected.", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    else:
        for cat, data in grouped_events.items():
            if data["is_global"]:
                cv2.putText(panel, f"[FILTERED] {cat}: Map-wide seasonal transition.", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            else:
                cv2.rectangle(panel, (15, y_offset - 12), (30, y_offset + 3), data["color"], -1)
                text = f"{cat.upper()}: {data['count']} distinct zones | Total Est. Area: {data['area']} px"
                cv2.putText(panel, text, (45, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 35 
    return np.vstack((map_img, panel))

def process_timeframe(d1, d2, layers, target_folder, visuals_dir, dataset_bounds):
    t1_imgs, t2_imgs = {}, {}
    for layer in layers:
        l_folder = os.path.join(target_folder, layer)
        img1 = load_layer_cd(d1, layer, l_folder)
        img2 = load_layer_cd(d2, layer, l_folder)
        if img1 is None or img2 is None: return None 
        t1_imgs[layer], t2_imgs[layer] = align_shapes(img1, img2)
        
    t1_rgb, t2_rgb = t1_imgs["true_color"], t2_imgs["true_color"]
    t1_nir, t2_nir = t1_imgs["false_color_nir"], t2_imgs["false_color_nir"]
    
    img_h, img_w = t1_rgb.shape[:2]
    img_shape = t1_rgb.shape
    events = []

    # Clean CLAHE equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g1 = clahe.apply(cv2.cvtColor(t1_rgb, cv2.COLOR_BGR2GRAY))
    g2 = clahe.apply(cv2.cvtColor(t2_rgb, cv2.COLOR_BGR2GRAY))
    
    _, diff = ssim(cv2.bilateralFilter(g1, 5, 50, 50), cv2.bilateralFilter(g2, 5, 50, 50), win_size=11, data_range=255, full=True)
    _, structure_change = cv2.threshold(cv2.bitwise_not((diff * 255).astype(np.uint8)), int((1 - SSIM_THRESH) * 255), 255, cv2.THRESH_BINARY)
    structure_change = cv2.morphologyEx(cv2.morphologyEx(structure_change, cv2.MORPH_OPEN, np.ones((3,3), np.uint8)), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    ndwi_t1 = calculate_ndwi(t1_rgb, t1_nir)
    ndwi_t2 = calculate_ndwi(t2_rgb, t2_nir)
    t1_w = np.uint8((ndwi_t1 > NDWI_WATER_THRESH) * 255)
    t2_w = np.uint8((ndwi_t2 > NDWI_WATER_THRESH) * 255)
    
    events = extract_change_clusters(cv2.bitwise_and(t2_w, cv2.bitwise_not(t1_w)), "Flooding", (255, 100, 100), events, img_shape) 
    events = extract_change_clusters(cv2.bitwise_and(t1_w, cv2.bitwise_not(t2_w)), "Water Receded", (0, 255, 255), events, img_shape) 

    ndvi_t1, ndvi_t2 = calculate_ndvi(t1_nir), calculate_ndvi(t2_nir)
    true_clearance_mask = cv2.bitwise_and(np.uint8(((ndvi_t1 > NDVI_VEG_THRESH) & ((ndvi_t1 - ndvi_t2) > NDVI_DROP_MIN)) * 255), structure_change)
    true_growth_mask = cv2.bitwise_and(np.uint8(((ndvi_t2 > NDVI_VEG_THRESH) & ((ndvi_t2 - ndvi_t1) > NDVI_DROP_MIN)) * 255), structure_change)

    diff_blur = cv2.bilateralFilter(cv2.subtract(cv2.cvtColor(t2_rgb, cv2.COLOR_BGR2GRAY), cv2.cvtColor(t1_rgb, cv2.COLOR_BGR2GRAY)), 5, 50, 50)
    _, gated_diff = cv2.threshold(diff_blur, 25, 255, cv2.THRESH_TOZERO)
    albedo_inc = cv2.threshold(gated_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] if np.max(gated_diff) > 0 else np.zeros_like(gated_diff)
    
    built_mask = cv2.bitwise_and(cv2.bitwise_and(albedo_inc, structure_change), cv2.bitwise_not(true_clearance_mask))

    total_pixels = img_h * img_w
    skip_clearance = (np.sum(true_clearance_mask == 255) / total_pixels) > GLOBAL_SEASONAL_LIMIT
    skip_growth = (np.sum(true_growth_mask == 255) / total_pixels) > GLOBAL_SEASONAL_LIMIT
    strict_construction = ((np.sum(built_mask == 255) / total_pixels) > GLOBAL_SEASONAL_LIMIT) or skip_clearance or skip_growth

    events = extract_change_clusters(true_clearance_mask, "Land Clearance", (0, 0, 255), events, img_shape, skip=skip_clearance) 
    events = extract_change_clusters(true_growth_mask, "Afforestation / Crop Growth", (0, 255, 0), events, img_shape, skip=skip_growth) 
    events = extract_change_clusters(built_mask, "Construction", (255, 0, 255), events, img_shape, check_linear=True, strict_mode=strict_construction)

    payload = {"timeframe": f"{d1}_to_{d2}", "features": []}
    
    # Generate the visual dashboard JPEG
    visual_proof = generate_dashboard_image(t2_rgb, events, d1, d2)
    cv2.imwrite(os.path.join(visuals_dir, f"Polygon_Dashboard_{d1}_to_{d2}.jpg"), visual_proof)
    
    for e in events:
        if e["is_global"] or e["contour"] is None: continue
        coord_list = []
        for point in e["contour"]:
            px, py = int(point[0][0]), int(point[0][1])
            coord_list.append(pixel_to_latlon(px, py, img_w, img_h, dataset_bounds) if dataset_bounds else [float(px), float(py)])
            
        if len(coord_list) >= 3:
            if coord_list[0] != coord_list[-1]: coord_list.append(coord_list[0])
            payload["features"].append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coord_list]},
                "properties": {"event_type": str(e["type"]), "area_pixels": int(e["area"]), "timeframe_start": str(d1), "timeframe_end": str(d2)}
            })
        
    del t1_imgs, t2_imgs, t1_rgb, t2_rgb, t1_nir, t2_nir, structure_change, built_mask, true_clearance_mask, true_growth_mask
    gc.collect()
    return payload

def run_automated_timeline(target_folder, available_dates):
    layers = ["true_color", "false_color_nir", "agriculture_swir"]
    visuals_dir = os.path.join(target_folder, "visual_reports")
    os.makedirs(visuals_dir, exist_ok=True)
    dataset_name = os.path.basename(target_folder)
    dataset_bounds = DATASET_BOUNDS.get(dataset_name, None)
    
    print("\n" + "=" * 70)
    print(f" ENTERPRISE TIMELINE PIPELINE INITIALIZED (Cleaned Data)")
    print("=" * 70)
    
    start_time = time.time()
    pairs = [(available_dates[i], available_dates[i+1]) for i in range(len(available_dates)-1)]
    master_geojson = {"type": "FeatureCollection", "name": f"Change_Detection_{dataset_name}", "features": []}

    safe_workers = max(1, min(4, os.cpu_count() or 4))
    completed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=safe_workers) as executor:
        future_to_pair = {executor.submit(process_timeframe, p[0], p[1], layers, target_folder, visuals_dir, dataset_bounds): p for p in pairs}
        for future in concurrent.futures.as_completed(future_to_pair):
            pair = future_to_pair[future]
            completed += 1
            print(f"\r[ {completed} / {len(pairs)} ] Processed: {pair[0]} ➔ {pair[1]}", end="", flush=True)
            try:
                result = future.result()
                if result: master_geojson["features"].extend(result["features"])
            except Exception as exc:
                print(f"\n[!] Error processing {pair[0]} ➔ {pair[1]}: {exc}")

    with open(os.path.join(visuals_dir, "timeline_analysis_data.geojson"), 'w') as f:
        json.dump(master_geojson, f, indent=4)

    print("\n" + "=" * 70)
    print(f"Pipeline Complete in {round(time.time() - start_time, 2)}s.")
    print(f"Visual Dashboards & GeoJSON Vectors saved in: \n{visuals_dir}")
    print("=" * 70)


# ==========================================
# CAPABILITY 1/4: SEMANTIC & VECTOR SEARCH
# ==========================================
def parse_date_from_filename(filename):
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    return datetime.strptime(match.group(0), "%Y-%m-%d") if match else None

def check_bbox_intersection(tile_bounds, search_aoi):
    if not tile_bounds or not search_aoi: return True
    t_min_lon, t_min_lat, t_max_lon, t_max_lat = tile_bounds
    s_min_lon, s_min_lat, s_max_lon, s_max_lat = search_aoi
    if t_max_lon < s_min_lon or t_min_lon > s_max_lon: return False
    if t_max_lat < s_min_lat or t_min_lat > s_max_lat: return False
    return True

def scan_dataset_folder_search(base_folder, layer_type="true_color"):
    valid_extensions = (".png", ".jpg", ".jpeg", ".tif")
    image_records = []
    
    keywords_map = {
        "true_color": ["true_color"],
        "false_color_nir": ["false_color_nir"],
        "agriculture_swir": ["agriculture_swir"]
    }
    target_keywords = keywords_map.get(layer_type, ["true_color"])

    for root, dirs, files in os.walk(base_folder):
        is_target_layer = any(term in root.lower() for term in target_keywords)
        if not is_target_layer: continue
        placename = os.path.basename(base_folder)
        bounds = DATASET_BOUNDS.get(placename, None)
        
        for f in files:
            if f.lower().endswith(valid_extensions):
                image_records.append({
                    "path": os.path.join(root, f),
                    "filename": f,
                    "placename": placename,
                    "bounds": bounds,
                    "date": parse_date_from_filename(f),
                    "layer_source": layer_type
                })
    return sorted(image_records, key=lambda x: x["filename"])

def load_and_preprocess(path):
    try: return preprocess(Image.open(path).convert("RGB"))
    except Exception: return None

@torch.no_grad()
def get_image_embeddings(image_paths, batch_size=32):
    if not image_paths: return np.array([])
    all_features = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(0, len(image_paths), batch_size):
            images_list = list(executor.map(load_and_preprocess, image_paths[i:i + batch_size]))
            valid_images = [img for img in images_list if img is not None]
            if not valid_images: continue
            
            images = torch.stack(valid_images).to(device)
            with torch.autocast(device_type=device.type):
                features = model.encode_image(images)
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().to(torch.float32).numpy())
            del images, features
            if device.type == 'cuda': torch.cuda.empty_cache()
    return np.vstack(all_features)

@torch.no_grad()
def get_ensemble_text_embedding(query_text, enhancements):
    prompts = [f"{query_text}, {enhancements}, high-resolution satellite imagery", f"satellite view of {query_text}"]
    text = tokenizer(prompts).to(device)
    with torch.autocast(device_type=device.type):
        features = model.encode_text(text)
    features /= features.norm(dim=-1, keepdim=True)
    mean_feature = features.mean(dim=0, keepdim=True)
    mean_feature /= mean_feature.norm(dim=-1, keepdim=True)
    return mean_feature.cpu().to(torch.float32).numpy()

def intelligent_layer_router(query):
    q = query.lower()
    layers, enhancements = set(), []
    if re.search(r'\b(construction|building|urban)\b', q):
        layers.update(["true_color", "false_color_nir"])
        enhancements.append("high albedo concrete, urban infrastructure")
    if re.search(r'\b(road|highway|street|bridge)\b', q):
        layers.update(["false_color_nir", "true_color"])
        enhancements.append("smooth thin linear asphalt trace")
    if re.search(r'\b(cleared|deforest|bare soil)\b', q):
        layers.add("false_color_nir")
        enhancements.append("exposed earth, deforestation")
    if re.search(r'\b(flood|water|river|lake)\b', q):
        layers.update(["agriculture_swir", "false_color_nir"])
        enhancements.append("pitch black pixel clusters, water absorbing infrared")
    if not layers:
        layers.update(["true_color"])
        enhancements.append("landscape overview")
    return list(layers), ", ".join(enhancements)

def execute_search(query_emb, records, embeddings, start_date=None, end_date=None, aoi=None):
    valid_indices = []
    for idx, rec in enumerate(records):
        if start_date and rec["date"] and rec["date"] < datetime.strptime(start_date, "%Y-%m-%d"): continue
        if end_date and rec["date"] and rec["date"] > datetime.strptime(end_date, "%Y-%m-%d"): continue
        if aoi and not check_bbox_intersection(rec["bounds"], aoi): continue
        valid_indices.append(idx)
        
    if not valid_indices: return []
    f_records = [records[i] for i in valid_indices]
    f_embeddings = embeddings[valid_indices]
    
    similarities = (f_embeddings @ query_emb.T).squeeze(-1) * 100
    ranked_indices = np.argsort(similarities)[::-1]
    
    artifact_text_features = get_artifact_text_features()
    logit_scale = model.logit_scale.exp().item()
    
    results, seen_paths = [], set()
    for idx in ranked_indices:
        if similarities[idx] < 0: break
        rec = f_records[idx]
        if rec["path"] in seen_paths: continue
        seen_paths.add(rec["path"])
        
        logits = (f_embeddings[idx] @ artifact_text_features.T) * logit_scale
        probs = softmax(logits)
        if probs[1] * 100 > 35.0 or probs[2] * 100 > 35.0: continue
            
        results.append({
            "placename": rec["placename"], "filename": rec["filename"], "path": rec["path"],
            "layer": rec["layer_source"], "score": round(float(similarities[idx]), 2),
            "date": rec["date"].strftime("%Y-%m-%d") if rec["date"] else None
        })
        if len(results) >= 15: break
    return results

def run_semantic_search(cleaned_folder):
    user_query = input("\nEnter intelligence requirement (e.g., 'newly built concrete structures'): ").strip()
    if not user_query: return
        
    target_layers, enhancements = intelligent_layer_router(user_query)
    print(f"\n[Intelligent Router] Searching Layers: {', '.join(target_layers)}")

    all_records, all_embeddings = [], []
    for layer in target_layers:
        records = scan_dataset_folder_search(cleaned_folder, layer)
        if not records: continue
        embs = get_image_embeddings([r["path"] for r in records])
        all_records.extend(records)
        all_embeddings.append(embs)
    
    if not all_records:
        print(" ❌ No tiles found in the cleaned dataset for those layers.")
        return
        
    combined_embeddings = np.vstack(all_embeddings)
    text_emb = get_ensemble_text_embedding(user_query, enhancements)
    results = execute_search(text_emb, all_records, combined_embeddings)
    
    print(f"\n--- RANKED GEOINT RESULTS FOR: '{user_query}' ---")
    if results:
        for idx, res in enumerate(results, 1):
            print(f"  {idx}. [Score: {res['score']}%] | Band: {res['layer']} | Date: {res['date']} | Tile: {res['filename']}")
    else:
        print("  ❌ No tiles matched the semantic requirements.")

def run_image_to_image_search(cleaned_folder):
    img_path = input("\nEnter filename or relative path of reference tile: ").strip().strip('"').strip("'")
    if not img_path: return
    
    found_ref = None
    ref_layer = "true_color"
    for layer in ["true_color", "false_color_nir", "agriculture_swir"]:
        recs = scan_dataset_folder_search(cleaned_folder, layer)
        match = next((r for r in recs if r["filename"] == os.path.basename(img_path) or os.path.abspath(r["path"]) == os.path.abspath(img_path)), None)
        if match:
            found_ref = match["path"]
            ref_layer = layer
            break
    
    if not found_ref or not os.path.exists(found_ref):
        print("  ❌ Error: Reference image tile could not be found locally across dataset folders.")
        return
        
    print(f"\n[Cross-Spectral Router] Found reference in layer band: '{ref_layer}'")
    records = scan_dataset_folder_search(cleaned_folder, ref_layer)
    
    embeddings = get_image_embeddings([r["path"] for r in records])
    ref_emb = get_image_embeddings([found_ref])
    
    results = execute_search(ref_emb, records, embeddings)
    results = [r for r in results if os.path.abspath(r["path"]) != os.path.abspath(found_ref)]
    
    print(f"\n--- RANKED CROSS-SPECTRAL MATCHES ---")
    if results:
        for idx, res in enumerate(results[:15], 1):
            print(f"  {idx}. [Score: {res['score']}%] | Band: {res['layer']} | Date: {res['date']} | Tile: {res['filename']}")
    else:
        print("  ❌ No matching cross-spectral neighbors found.")


# ==========================================
# UNIFIED MASTER MENU
# ==========================================
def main_menu():
    raw_base_dir = "./satellite_datasets"
    cleaned_base_dir = "./satellite_datasets_cleaned"

    if not os.path.exists(raw_base_dir):
        print(f"Error: Base directory '{raw_base_dir}' not found.")
        return

    folders = [f for f in os.listdir(raw_base_dir) if os.path.isdir(os.path.join(raw_base_dir, f))]
    if not folders:
        print(f"No datasets found in '{raw_base_dir}'.")
        return

    print("=" * 60)
    print(" UNIFIED MASTER GEOINT DASHBOARD")
    print("=" * 60)
    print("Available Raw Datasets:")
    for i, folder in enumerate(folders, 1):
        print(f"   [{i}] {folder}")
    
    try:
        folder_choice = int(input("\nSelect a dataset region to load: ").strip()) - 1
        if folder_choice < 0 or folder_choice >= len(folders): return
    except ValueError: return
        
    region_name = folders[folder_choice]
    cleaned_region_path = os.path.join(cleaned_base_dir, region_name)

    needs_cleaning = True
    if os.path.exists(cleaned_region_path) and os.listdir(cleaned_region_path):
        ans = input(f"\nCleaned data already exists for '{region_name}'. Re-run Quality Gating & Cleaning? (y/n): ").strip().lower()
        if ans != 'y':
            needs_cleaning = False

    if needs_cleaning:
        print(f"\n[Phase 1] Running Capability 3 Quality Gating & Standardization on '{region_name}'...")
        cleaned_region_path = run_quality_pipeline(region_name, base_dir=raw_base_dir)

    while True:
        print("\n" + "=" * 60)
        print(f" ACTIVE REGION: {region_name} (Using Cleaned Data)")
        print("=" * 60)
        print("  [1] Run GIS Change Detection & Timeline Dashboard (Cap 2)")
        print("  [2] Plain English / Semantic Text Search (Cap 1)")
        print("  [3] Image-to-Image Cross-Spectral Search (Find Similar Trouble Spots) (Cap 4)")
        print("  [4] Exit")
        print("=" * 60)
        
        choice = input("Select feature to run (1, 2, 3, or 4): ").strip()
        if choice == '4':
            break

        elif choice == '1':
            tc_dir = os.path.join(cleaned_region_path, "true_color")
            if not os.path.exists(tc_dir):
                print("Error: Missing 'true_color' layer in cleaned directory.")
                continue
                
            files = os.listdir(tc_dir)
            available_dates = sorted(list({re.search(r"(\d{4}-\d{2}-\d{2})", f).group(1) for f in files if re.search(r"(\d{4}-\d{2}-\d{2})", f)}))
            
            if len(available_dates) < 2:
                print("Not enough valid dates remaining after quality gating to build a timeline.")
                continue

            print(f"\n[Phase 2] Locked {len(available_dates)} clean dates. Running Change Detection...")
            run_automated_timeline(cleaned_region_path, available_dates)

        elif choice == '2':
            print(f"\n[Phase 2] Running Semantic Vector Search on clean data...")
            run_semantic_search(cleaned_region_path)

        elif choice == '3':
            print(f"\n[Phase 2] Running Image-to-Image Cross-Spectral Search on clean data...")
            run_image_to_image_search(cleaned_region_path)
            
        else:
            print("Invalid selection. Please choose 1, 2, 3, or 4.")

if __name__ == "__main__":
    main_menu()
  
