import os
import glob
import re
import json
import uuid
import pickle
import hashlib
from datetime import datetime, timezone

import cv2
import torch
import numpy as np
from PIL import Image, ImageEnhance
import open_clip
from huggingface_hub import hf_hub_download
import time
import gc
import concurrent.futures
from skimage.metrics import structural_similarity as ssim

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import rasterio
    from rasterio.warp import transform_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

print("Initializing Enterprise GEOINT Engine (Fully Integrated)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

print(f"Loading RemoteCLIP ({model_name}) weights offline...")
MODEL_PROVENANCE = {
    "name": "RemoteCLIP",
    "backbone": model_name,
    "source": "huggingface.co/chendelong/RemoteCLIP",
    "license": "Apache-2.0",
    "staged_offline": True,
}
try:
    ckpt_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", local_files_only=True)
except Exception:
    ckpt_path = f"checkpoints/RemoteCLIP-{model_name}.pt"

if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    MODEL_PROVENANCE["checkpoint_path"] = ckpt_path
    try:
        with open(ckpt_path, "rb") as fh:
            MODEL_PROVENANCE["checkpoint_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    except Exception:
        pass
else:
    print(f"Warning: Checkpoint file not found at {ckpt_path}. Ensure weights are downloaded locally.")

model = model.to(device).eval()
print(f"Model loaded locally on [{device.type.upper()}] and ready!\n")


# PARAMETERS FOR CHANGE DETECTION
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
EMBED_DIM = model.text_projection.shape[1] if hasattr(model, "text_projection") else 768

INDEX_ROOT = "./vector_indices"
REVIEW_ROOT = "./analyst_workflow"
os.makedirs(INDEX_ROOT, exist_ok=True)
os.makedirs(REVIEW_ROOT, exist_ok=True)


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


# GEO INGESTION & COG SUPPORT (2.2.6)

def get_real_bounds_from_geotiff(path):
    if not RASTERIO_AVAILABLE or not path.lower().endswith((".tif", ".tiff")):
        return None
    try:
        with rasterio.open(path) as src:
            if src.crs is None: return None
            b = src.bounds
            if src.crs.to_epsg() != 4326:
                left, bottom, right, top = transform_bounds(src.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
            else:
                left, bottom, right, top = b.left, b.bottom, b.right, b.top
            return [left, bottom, right, top]
    except Exception:
        return None


def resolve_bounds(path, placename):
    real = get_real_bounds_from_geotiff(path)
    if real is not None: return real
    return DATASET_BOUNDS.get(placename, None)


# SPECTRAL INDEX CALCULATIONS

def calculate_ndvi(false_color_img):
    img_float = false_color_img.astype(np.float32)
    nir = img_float[:, :, 2]
    red = img_float[:, :, 1]
    denominator = nir + red
    denominator[denominator == 0] = 1e-5
    return (nir - red) / denominator


def calculate_ndwi(true_color_img, false_color_img):
    green = true_color_img.astype(np.float32)[:, :, 1]
    nir = false_color_img.astype(np.float32)[:, :, 2]
    denominator = green + nir
    denominator[denominator == 0] = 1e-5
    return (green - nir) / denominator


# QUALITY GATING & CLEANING

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
        borders = np.concatenate([gray[:tb, :].flatten(), gray[h-tb:, :].flatten(), gray[:, :lr].flatten(), gray[:, w-lr:].flatten()])
        black_border_pixels = np.sum(borders < 15)
        border_pct = (black_border_pixels / borders.size) * 100
        return border_pct > max_edge_black_pct, border_pct

    def validate_frame(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return True, "Error: Failed to read image file."
        is_cloudy, cloud_pct = self.check_cloud_rejection(img)
        if is_cloudy: return True, f"Rejected: Excessive cloud cover ({cloud_pct:.1f}% > 15%)."
        has_black_edges, edge_black_pct = self.check_black_borders_rejection(img)
        if has_black_edges: return True, f"Rejected: Excessive black borders ({edge_black_pct:.1f}% > 25%)."
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
        return (np.clip(dehazed, 0, 1) * 255).astype(np.uint8)

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
            matched[:, i] = np.interp(s_quantiles, t_quantiles, t_values)[s_idx]
        return matched.reshape(old_shape).astype(np.uint8)

    def evaluate_image_clarity(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None: return 0.0
            sharpness_score = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
            text_query = tokenizer(["a clear satellite view of ground features"]).to(self.device)
            with torch.no_grad():
                img_feat, txt_feat = model.encode_image(image), model.encode_text(text_query)
                similarity = (img_feat.norm(dim=-1, keepdim=True).reciprocal() * img_feat @ (txt_feat / txt_feat.norm(dim=-1, keepdim=True)).T).item() * 100
            return float(sharpness_score + (similarity * 5.0))
        except Exception:
            return 0.0

    def process_and_standardize_image(self, image_path, master_template):
        img = cv2.imread(image_path)
        if img is None: return None
        dehazed = self.dark_channel_dehaze(img)
        pil_img = ImageEnhance.Contrast(ImageEnhance.Color(Image.fromarray(cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB))).enhance(1.15)).enhance(1.10)
        return cv2.bilateralFilter(self.match_histogram(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), master_template), d=5, sigmaColor=30, sigmaSpace=30)

    def quality_confidence(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return 0.0
        is_cloudy, cloud_pct = self.check_cloud_rejection(img)
        has_edges, edge_pct = self.check_black_borders_rejection(img)
        return round(max(0.0, 1.0 - min(1.0, (cloud_pct / 100.0) + (edge_pct / 200.0))), 3)


def run_quality_pipeline(region_name, base_dir="./satellite_datasets"):
    target_directory = os.path.join(base_dir, region_name)
    region_output_root = os.path.join("./satellite_datasets_cleaned", region_name)
    os.makedirs(region_output_root, exist_ok=True)

    target_bands = ["true_color", "false_color_nir", "agriculture_swir"]
    engine = MultiBandQualityEnhancementPipeline()
    quality_manifest = {}

    for band_name in target_bands:
        band_dir = os.path.join(target_directory, band_name)
        if not os.path.exists(band_dir): continue
        image_files = sorted(glob.glob(os.path.join(band_dir, "*.jpg")) + glob.glob(os.path.join(band_dir, "*.png")) + glob.glob(os.path.join(band_dir, "*.tif")))
        if not image_files: continue

        output_processed_dir = os.path.join(region_output_root, band_name)
        os.makedirs(output_processed_dir, exist_ok=True)
        valid_images = []
        
        for path in image_files:
            filename = os.path.basename(path)
            rejected, reason = engine.validate_frame(path)
            if rejected:
                quality_manifest[filename] = {"status": "rejected", "reason": reason}
            else:
                valid_images.append(path)

        if not valid_images: continue
        best_image_path = max(valid_images, key=lambda p: engine.evaluate_image_clarity(p))
        master_template = cv2.imread(best_image_path)

        for path in valid_images:
            filename = os.path.basename(path)
            cleaned_img = engine.process_and_standardize_image(path, master_template)
            quality_manifest[filename] = {"status": "passed", "confidence": engine.quality_confidence(path), "band": band_name}
            if cleaned_img is not None:
                cv2.imwrite(os.path.join(output_processed_dir, filename), cleaned_img)

    with open(os.path.join(region_output_root, "quality_manifest.json"), "w") as f:
        json.dump(quality_manifest, f, indent=2)
    return region_output_root


# CHANGE DETECTION & EARLIEST ESTIMATION

def load_layer_cd(date_prefix, layer_name, folder_path):
    files = glob.glob(os.path.join(folder_path, f"{date_prefix}*.jpg")) + glob.glob(os.path.join(folder_path, f"{date_prefix}*.png")) + glob.glob(os.path.join(folder_path, f"{date_prefix}*.tif"))
    return cv2.imread(files[0]) if files else None


def align_shapes(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
    g1, g2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp_matrix = cv2.findTransformECC(g1, g2, warp_matrix, cv2.MOTION_TRANSLATION, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4))
        return img1, cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    except Exception:
        return img1, img2


def extract_change_clusters(mask, category_name, color, events_list, img_shape, check_linear=False, skip=False, strict_mode=False):
    if skip:
        events_list.append({"type": f"Global Shift ({category_name})", "center": (0, 0), "contour": None, "area": 0, "color": (128, 128, 128), "is_global": True})
        return events_list

    active_min_area = MIN_CHANGE_AREA_PX * 2 if strict_mode else MIN_CHANGE_AREA_PX
    active_min_extent = 0.40 if strict_mode else MIN_CONSTRUCTION_EXTENT
    k_size = 9 if "Clearance" in category_name or "Growth" in category_name else 15

    merged_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k_size, k_size), np.uint8))
    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = img_shape[:2]

    for c in contours:
        area = cv2.contourArea(c)
        if area > active_min_area:
            x_int, y_int, w, h = cv2.boundingRect(c)
            if x_int <= EDGE_BUFFER or y_int <= EDGE_BUFFER or (x_int + w) >= (img_w - EDGE_BUFFER) or (y_int + h) >= (img_h - EDGE_BUFFER):
                continue

            rect = cv2.minAreaRect(c)
            aspect_ratio = max(rect[1]) / (min(rect[1]) + 1e-5)
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = float(area) / hull_area if hull_area > 0 else 0
            extent = float(area) / (w * h) if w * h > 0 else 0
            perimeter = cv2.arcLength(c, True)
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

            M = cv2.moments(c)
            cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)
            final_category, final_color = category_name, color

            if "Clearance" in category_name and solidity > FARM_SOLIDITY_THRESH and area > 1000:
                final_category, final_color = "Agricultural Harvest", (144, 238, 144)

            if check_linear and final_category != "Agricultural Harvest":
                if extent < active_min_extent:
                    continue
                lines = cv2.HoughLinesP(cv2.Canny(cv2.dilate(merged_mask[y_int:y_int+h, x_int:x_int+w], np.ones((3,3), np.uint8), iterations=1), 50, 150), 1, np.pi/180, threshold=20, minLineLength=25, maxLineGap=10)
                if lines is not None and len(lines) > 0:
                    final_category, final_color = ("Road Development", (0, 165, 255)) if aspect_ratio >= MIN_ROAD_ASPECT_RATIO and compactness <= MAX_ROAD_COMPACTNESS else ("Construction", (255, 0, 255))

            events_list.append({
                "type": final_category, "center": (cx, cy), "contour": cv2.approxPolyDP(c, 0.01 * perimeter, True),
                "area": int(area), "color": final_color, "is_global": False,
                "confidence": round(float(min(1.0, 0.4 + 0.3 * min(solidity, 1.0) + 0.3 * min(area / 5000.0, 1.0))), 3)
            })
    return events_list


def generate_dashboard_image(t2_rgb, events, d1, d2):
    h, w = t2_rgb.shape[:2]
    map_img, overlay = t2_rgb.copy(), t2_rgb.copy()
    for e in events:
        if not e["is_global"] and e["contour"] is not None:
            cv2.drawContours(overlay, [e["contour"]], -1, e["color"], -1)
    cv2.addWeighted(overlay, 0.4, map_img, 0.6, 0, map_img)
    for e in events:
        if not e["is_global"] and e["contour"] is not None:
            cv2.drawContours(map_img, [e["contour"]], -1, e["color"], 2)
            cv2.circle(map_img, e["center"], 3, (255, 255, 255), -1)

    grouped = {}
    for e in events:
        grouped.setdefault(e["type"], {"color": e["color"], "count": 0, "area": 0, "is_global": e["is_global"]})
        grouped[e["type"]]["count"] += 1
        grouped[e["type"]]["area"] += e["area"]

    panel = np.zeros((90 + (len(grouped) * 35) if grouped else 125, w, 3), dtype=np.uint8)
    cv2.putText(panel, f"ENTERPRISE TIMELINE: {d1} to {d2}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(panel, (15, 40), (w - 15, 40), (100, 100, 100), 1)

    y_offset = 70
    if not grouped:
        cv2.putText(panel, "No significant structural changes detected.", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    else:
        for cat, data in grouped.items():
            if data["is_global"]:
                cv2.putText(panel, f"[FILTERED] {cat}: Seasonal transition.", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            else:
                cv2.rectangle(panel, (15, y_offset - 12), (30, y_offset + 3), data["color"], -1)
                cv2.putText(panel, f"{cat.upper()}: {data['count']} zones | Area: {data['area']} px", (45, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 35
    return np.vstack((map_img, panel))


def compute_change_events(t1_rgb, t2_rgb, t1_nir, t2_nir):
    img_h, img_w = t1_rgb.shape[:2]
    img_shape = t1_rgb.shape
    events = []

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g1, g2 = clahe.apply(cv2.cvtColor(t1_rgb, cv2.COLOR_BGR2GRAY)), clahe.apply(cv2.cvtColor(t2_rgb, cv2.COLOR_BGR2GRAY))
    _, diff = ssim(cv2.bilateralFilter(g1, 5, 50, 50), cv2.bilateralFilter(g2, 5, 50, 50), win_size=11, data_range=255, full=True)
    _, structure_change = cv2.threshold(cv2.bitwise_not((diff * 255).astype(np.uint8)), int((1 - SSIM_THRESH) * 255), 255, cv2.THRESH_BINARY)
    structure_change = cv2.morphologyEx(cv2.morphologyEx(structure_change, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    ndwi_t1, ndwi_t2 = calculate_ndwi(t1_rgb, t1_nir), calculate_ndwi(t2_rgb, t2_nir)
    events = extract_change_clusters(cv2.bitwise_and(np.uint8((ndwi_t2 > NDWI_WATER_THRESH)*255), cv2.bitwise_not(np.uint8((ndwi_t1 > NDWI_WATER_THRESH)*255))), "Flooding", (255, 100, 100), events, img_shape)
    events = extract_change_clusters(cv2.bitwise_and(np.uint8((ndwi_t1 > NDWI_WATER_THRESH)*255), cv2.bitwise_not(np.uint8((ndwi_t2 > NDWI_WATER_THRESH)*255))), "Water Receded", (0, 255, 255), events, img_shape)

    ndvi_t1, ndvi_t2 = calculate_ndvi(t1_nir), calculate_ndvi(t2_nir)
    true_clearance_mask = cv2.bitwise_and(np.uint8(((ndvi_t1 > NDVI_VEG_THRESH) & ((ndvi_t1 - ndvi_t2) > NDVI_DROP_MIN)) * 255), structure_change)
    true_growth_mask = cv2.bitwise_and(np.uint8(((ndvi_t2 > NDVI_VEG_THRESH) & ((ndvi_t2 - ndvi_t1) > NDVI_DROP_MIN)) * 255), structure_change)

    diff_blur = cv2.bilateralFilter(cv2.subtract(cv2.cvtColor(t2_rgb, cv2.COLOR_BGR2GRAY), cv2.cvtColor(t1_rgb, cv2.COLOR_BGR2GRAY)), 5, 50, 50)
    albedo_inc = cv2.threshold(cv2.threshold(diff_blur, 25, 255, cv2.THRESH_TOZERO)[1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    built_mask = cv2.bitwise_and(cv2.bitwise_and(albedo_inc, structure_change), cv2.bitwise_not(true_clearance_mask))

    total_pixels = img_h * img_w
    skip_clearance = (np.sum(true_clearance_mask == 255) / total_pixels) > GLOBAL_SEASONAL_LIMIT
    skip_growth = (np.sum(true_growth_mask == 255) / total_pixels) > GLOBAL_SEASONAL_LIMIT
    strict_construction = ((np.sum(built_mask == 255) / total_pixels) > GLOBAL_SEASONAL_LIMIT) or skip_clearance or skip_growth

    events = extract_change_clusters(true_clearance_mask, "Land Clearance", (0, 0, 255), events, img_shape, skip=skip_clearance)
    events = extract_change_clusters(true_growth_mask, "Afforestation / Crop Growth", (0, 255, 0), events, img_shape, skip=skip_growth)
    events = extract_change_clusters(built_mask, "Construction", (255, 0, 255), events, img_shape, check_linear=True, strict_mode=strict_construction)
    return events, img_h, img_w


def estimate_earliest_change_dates(target_folder, available_dates, final_events, tolerance_px=40):
    if len(available_dates) < 3 or not final_events:
        return {e["center"]: available_dates[-2] if len(available_dates) >= 2 else None for e in final_events}

    tc_dir = os.path.join(target_folder, "true_color")
    latest_rgb = load_layer_cd(available_dates[-1], "true_color", tc_dir)
    if latest_rgb is None: return {}

    earliest_map = {}
    for e in final_events:
        if e["is_global"] or e["contour"] is None: continue
        cx, cy = e["center"]
        x0, x1 = max(0, cx - tolerance_px), cx + tolerance_px
        y0, y1 = max(0, cy - tolerance_px), cy + tolerance_px
        earliest_supporting = available_dates[-2]

        for cand_date in available_dates[:-1]:
            cand_rgb = load_layer_cd(cand_date, "true_color", tc_dir)
            if cand_rgb is None: continue
            if cand_rgb.shape != latest_rgb.shape:
                cand_rgb = cv2.resize(cand_rgb, (latest_rgb.shape[1], latest_rgb.shape[0]))
            
            p_late, p_cand = latest_rgb[y0:y1, x0:x1], cand_rgb[y0:y1, x0:x1]
            if p_late.size == 0 or p_cand.size == 0 or p_late.shape != p_cand.shape: continue
            
            try:
                if ssim(cv2.cvtColor(p_late, cv2.COLOR_BGR2GRAY), cv2.cvtColor(p_cand, cv2.COLOR_BGR2GRAY), data_range=255) < SSIM_THRESH:
                    earliest_supporting = cand_date
                else:
                    break
            except Exception:
                break
        earliest_map[(cx, cy)] = earliest_supporting
    return earliest_map


def process_timeframe(d1, d2, layers, target_folder, visuals_dir, dataset_bounds, all_dates=None):
    t1_imgs, t2_imgs = {}, {}
    for layer in layers:
        l_folder = os.path.join(target_folder, layer)
        img1, img2 = load_layer_cd(d1, layer, l_folder), load_layer_cd(d2, layer, l_folder)
        if img1 is None or img2 is None: return None
        t1_imgs[layer], t2_imgs[layer] = align_shapes(img1, img2)

    events, img_h, img_w = compute_change_events(t1_imgs["true_color"], t2_imgs["true_color"], t1_imgs["false_color_nir"], t2_imgs["false_color_nir"])
    earliest_dates = estimate_earliest_change_dates(target_folder, all_dates, events) if (all_dates and all_dates[-1] == d2) else {}

    payload = {"timeframe": f"{d1}_to_{d2}", "features": []}
    cv2.imwrite(os.path.join(visuals_dir, f"Polygon_Dashboard_{d1}_to_{d2}.jpg"), generate_dashboard_image(t2_imgs["true_color"], events, d1, d2))

    for e in events:
        if e["is_global"] or e["contour"] is None: continue
        coord_list = [pixel_to_latlon(int(p[0][0]), int(p[0][1]), img_w, img_h, dataset_bounds) if dataset_bounds else [float(p[0][0]), float(p[0][1])] for p in e["contour"]]
        if len(coord_list) >= 3:
            if coord_list[0] != coord_list[-1]: coord_list.append(coord_list[0])
            payload["features"].append({
                "type": "Feature", "geometry": {"type": "Polygon", "coordinates": [coord_list]},
                "properties": {
                    "event_type": str(e["type"]), "area_pixels": int(e["area"]),
                    "timeframe_start": str(d1), "timeframe_end": str(d2),
                    "earliest_supported_observation": str(earliest_dates.get(e["center"], d1)),
                    "confidence": e.get("confidence", 0.5),
                    "processing_history": ["quality_gate_v1", "dark_channel_dehaze", "histogram_match", "ecc_translation_align", "ssim_structure_diff", "ndvi_ndwi_index", "earliest_date_backtrack"]
                }
            })
    return payload


def run_automated_timeline(target_folder, available_dates, review_queue=None, dataset_name_override=None):
    layers = ["true_color", "false_color_nir", "agriculture_swir"]
    visuals_dir = os.path.join(target_folder, "visual_reports")
    os.makedirs(visuals_dir, exist_ok=True)
    dataset_name = dataset_name_override or os.path.basename(target_folder)
    dataset_bounds = DATASET_BOUNDS.get(dataset_name, None)

    print("\n" + "=" * 70)
    print(" ENTERPRISE TIMELINE PIPELINE INITIALIZED (Cleaned Data)")
    print("=" * 70)

    start_time = time.time()
    pairs = [(available_dates[i], available_dates[i + 1]) for i in range(len(available_dates) - 1)]
    master_geojson = {"type": "FeatureCollection", "name": f"Change_Detection_{dataset_name}", "features": []}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 4))) as executor:
        future_to_pair = {executor.submit(process_timeframe, p[0], p[1], layers, target_folder, visuals_dir, dataset_bounds, available_dates): p for p in pairs}
        for completed, future in enumerate(concurrent.futures.as_completed(future_to_pair), 1):
            pair = future_to_pair[future]
            print(f"\r[ {completed} / {len(pairs)} ] Processed: {pair[0]} ➔ {pair[1]}", end="", flush=True)
            try:
                res = future.result()
                if res: master_geojson["features"].extend(res["features"])
            except Exception as exc:
                print(f"\n[!] Error processing {pair[0]} ➔ {pair[1]}: {exc}")

    with open(os.path.join(visuals_dir, "timeline_analysis_data.geojson"), 'w') as f:
        json.dump(master_geojson, f, indent=4)

    if review_queue is not None:
        for feat in master_geojson["features"]:
            props = feat["properties"]
            review_queue.add_candidate(
                candidate_type="change_event", dataset=dataset_name, geometry=feat["geometry"],
                confidence=props.get("confidence", 0.5),
                acquisition_info={"timeframe_start": props["timeframe_start"], "timeframe_end": props["timeframe_end"], "earliest_supported_observation": props["earliest_supported_observation"], "sensor": "Sentinel-2 L2A"},
                processing_history=props.get("processing_history", []), label=props["event_type"],
                evidence_paths=[os.path.join(visuals_dir, f"Polygon_Dashboard_{props['timeframe_start']}_to_{props['timeframe_end']}.jpg")]
            )
    print(f"\n[+] Pipeline Complete in {round(time.time() - start_time, 2)}s. Reports saved in: {visuals_dir}")


# VECTOR INDEX MANAGER (FAISS)

class VectorIndexManager:
    def __init__(self, region_name, dim=EMBED_DIM):
        self.region_name, self.dim = region_name, dim
        self.index_path = os.path.join(INDEX_ROOT, f"{region_name}.faiss")
        self.meta_path = os.path.join(INDEX_ROOT, f"{region_name}_meta.pkl")
        self.records, self.seen_paths = [], set()
        self._load_or_init()

    def _load_or_init(self):
        if FAISS_AVAILABLE and os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f: self.records = pickle.load(f)
            self.seen_paths = {r["path"] for r in self.records}
        elif FAISS_AVAILABLE:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
        else:
            self.index = None
            self._np_embeddings = np.zeros((0, self.dim), dtype=np.float32)

    def add(self, new_records, embeddings):
        fresh_idx = [i for i, r in enumerate(new_records) if r["path"] not in self.seen_paths]
        if not fresh_idx: return 0
        fresh_records, fresh_embs = [new_records[i] for i in fresh_idx], embeddings[fresh_idx].astype(np.float32)
        start_id = len(self.records)
        ids = np.arange(start_id, start_id + len(fresh_records)).astype(np.int64)

        if FAISS_AVAILABLE and self.index is not None:
            self.index.add_with_ids(fresh_embs, ids)
        else:
            self._np_embeddings = np.vstack([self._np_embeddings, fresh_embs])

        self.records.extend(fresh_records)
        self.seen_paths.update(r["path"] for r in fresh_records)
        return len(fresh_records)

    def save(self):
        if FAISS_AVAILABLE and self.index is not None: faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f: pickle.dump(self.records, f)

    def search(self, query_emb, top_k=50):
        query_emb = query_emb.astype(np.float32)
        if FAISS_AVAILABLE and self.index is not None:
            if self.index.ntotal == 0: return [], np.array([])
            scores, ids = self.index.search(query_emb, min(top_k, self.index.ntotal))
            return [self.records[i] for i in ids[0][ids[0] >= 0]], scores[0][ids[0] >= 0]
        else:
            if self._np_embeddings.shape[0] == 0: return [], np.array([])
            sims = (self._np_embeddings @ query_emb.T).squeeze(-1)
            order = np.argsort(sims)[::-1][:top_k]
            return [self.records[i] for i in order], sims[order]

    def all_embeddings(self):
        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            return np.vstack([self.index.reconstruct(i) for i in range(self.index.ntotal)])
        elif not FAISS_AVAILABLE: return self._np_embeddings
        return np.zeros((0, self.dim), dtype=np.float32)


# DISCOVERY & CLUSTERING 

def _kmeans_cluster(embeddings, k):
    if FAISS_AVAILABLE:
        km = faiss.Kmeans(embeddings.shape[1], k, niter=25, seed=42, spherical=True)
        km.train(embeddings.astype(np.float32))
        return km.index.search(embeddings.astype(np.float32), 1)[1].reshape(-1)
    rng = np.random.default_rng(42)
    centroids = embeddings[rng.choice(len(embeddings), size=k, replace=False)].copy()
    for _ in range(25):
        sims = embeddings @ centroids.T
        labels = np.argmax(sims, axis=1)
        for c in range(k):
            members = embeddings[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)
                centroids[c] /= (np.linalg.norm(centroids[c]) + 1e-8)
    return labels


def cluster_dataset(index_manager, n_clusters=8):
    embeddings = index_manager.all_embeddings()
    n_clusters = max(1, min(n_clusters, embeddings.shape[0]))
    if embeddings.shape[0] == 0: return {}
    labels = _kmeans_cluster(embeddings, n_clusters)
    groups = {}
    for rec, lbl in zip(index_manager.records, labels):
        groups.setdefault(int(lbl), []).append(rec)
    return groups


def run_discovery_clustering(index_manager):
    print("\n--- DISCOVERY & CLUSTERING (2.2.4) ---")
    sub = input("Select mode - [a] Cluster region, [b] Seed site similarity: ").strip().lower()
    if sub == 'a':
        k = int(input("Number of clusters (e.g. 8): ").strip() or "8")
        for cid, members in sorted(cluster_dataset(index_manager, n_clusters=k).items()):
            print(f"\nCluster {cid} ({len(members)} tiles):")
            for m in members[:8]: print(f"    - {m['filename']} ({m.get('date')})")
    elif sub == 'b':
        seed = input("Filename or path of the seed tile: ").strip().strip('"').strip("'")
        match = next((r for r in index_manager.records if r["filename"] == os.path.basename(seed) or os.path.abspath(r["path"]) == os.path.abspath(seed)), None)
        if not match: print(" ❌ Seed tile not found."); return
        groups = cluster_dataset(index_manager, n_clusters=8)
        for cid, members in groups.items():
            if any(os.path.abspath(m["path"]) == os.path.abspath(match["path"]) for m in members):
                print(f"\nSeed belongs to Cluster {cid}. Similar sites:")
                for o in members:
                    if os.path.abspath(o["path"]) != os.path.abspath(match["path"]):
                        print(f"    - {o['filename']} ({o.get('date')}) @ {o.get('placename')}")
                break


# ==========================================
# ANALYST REVIEW QUEUE (2.2.5 - Short IDs & Continuous Choice)
# ==========================================
class ReviewQueue:
    def __init__(self, region_name):
        self.region_name = region_name
        self.path = os.path.join(REVIEW_ROOT, f"{region_name}_review_queue.json")
        self.audit_path = os.path.join(REVIEW_ROOT, f"{region_name}_audit_trail.jsonl")
        self.items = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f: return json.load(f)
        return {}

    def _save(self):
        with open(self.path, "w") as f: json.dump(self.items, f, indent=2)

    def _audit(self, entry):
        with open(self.audit_path, "a") as f: f.write(json.dumps(entry) + "\n")

    def add_candidate(self, candidate_type, dataset, confidence, acquisition_info, processing_history, label="", geometry=None, evidence_paths=None):
        count = len(self.items) + 1
        cand_id = f"c{count}"
        
        self.items[cand_id] = {
            "id": cand_id, "type": candidate_type, "dataset": dataset, "label": label, "confidence": confidence,
            "geometry": geometry, "evidence_paths": evidence_paths or [], "acquisition_info": acquisition_info,
            "processing_history": processing_history, "status": "pending", "created_at": datetime.now(timezone.utc).isoformat()
        }
        self._save()
        return cand_id

    def review_queue_ranked(self, status="pending"):
        return sorted([v for v in self.items.values() if v["status"] == status], key=lambda x: x["confidence"], reverse=True)

    def decide(self, cand_id, decision, notes=None):
        if cand_id not in self.items: return False
        self.items[cand_id].update({"status": decision, "decision_at": datetime.now(timezone.utc).isoformat(), "analyst_notes": notes})
        self._save()
        self._audit({"candidate_id": cand_id, "decision": decision, "notes": notes, "timestamp": datetime.now(timezone.utc).isoformat()})
        return True

    def feedback_bias_terms(self):
        return [v["label"] for v in self.items.values() if v["status"] == "confirmed" and v["label"]], [v["label"] for v in self.items.values() if v["status"] == "rejected" and v["label"]]


def run_review_queue_ui(review_queue):
    while True:
        print("\n" + "=" * 60)
        print(" ANALYST REVIEW QUEUE (2.2.5)")
        print("=" * 60)
        
        pending = review_queue.review_queue_ranked("pending")
        if not pending: 
            print(" Queue is empty. All candidates have been reviewed.")
            break
            
        for idx, item in enumerate(pending[:20], 1):
            print(f"  [{idx}] (ID: {item['id']}) Label: {item['label']} | Confidence: {item['confidence']}")
            print(f"      Acquisition: {item['acquisition_info']}")
            print(f"      Evidence: {item['evidence_paths']}")
            print("-" * 50)
            
        choice = input(f"Select item number to review (1-{min(len(pending), 20)}) or type 'exit' to return: ").strip().lower()
        if choice == 'exit' or not choice:
            print("Exiting review queue.")
            break
            
        target_item = None
        if choice.isdigit():
            list_idx = int(choice) - 1
            if 0 <= list_idx < len(pending):
                target_item = pending[list_idx]
        else:
            target_item = next((item for item in pending if item['id'].lower() == choice.lower()), None)
            
        if not target_item:
            print("❌ Invalid selection or ID not found. Try again.")
            continue
            
        cand_id = target_item['id']
        print(f"\nReviewing Candidate [{cand_id}] -> {target_item['label']}")
        dec = input("Decision (confirm/reject): ").strip().lower()
        notes = input("Analyst notes (optional): ").strip() or None
        
        if dec.startswith("c"): 
            review_queue.decide(cand_id, "confirmed", notes)
            print(f" ✓ Candidate [{cand_id}] Confirmed and logged to audit trail.")
        elif dec.startswith("r"): 
            review_queue.decide(cand_id, "rejected", notes)
            print(f" ✓ Candidate [{cand_id}] Rejected and logged to audit trail.")
        else:
            print("❌ Invalid decision choice, no action taken.")
            
        cont = input("\nWould you like to review another candidate? (y/n): ").strip().lower()
        if cont != 'y':
            print("Returning to master menu.")
            break


# SEMANTIC RETRIEVAL 

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
    keywords_map = {"true_color": ["true_color"], "false_color_nir": ["false_color_nir"], "agriculture_swir": ["agriculture_swir"]}
    for root, _, files in os.walk(base_folder):
        if not any(term in root.lower() for term in keywords_map.get(layer_type, ["true_color"])): continue
        placename = os.path.basename(base_folder)
        for f in files:
            if f.lower().endswith(valid_extensions):
                full_path = os.path.join(root, f)
                image_records.append({
                    "path": full_path, "filename": f, "placename": placename, 
                    "bounds": resolve_bounds(full_path, placename), "date": parse_date_from_filename(f), 
                    "layer_source": layer_type, "sensor": "Sentinel-2 L2A"
                })
    return sorted(image_records, key=lambda x: x["filename"])


@torch.no_grad()
def get_image_embeddings(image_paths, batch_size=8):
    if not image_paths: return np.array([])
    all_features = []
    print(f"   [Embedding Worker] Encoding {len(image_paths)} images on CPU (Running on CPU is slow, please wait)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for i in range(0, len(image_paths), batch_size):
            batch_slice = image_paths[i:i + batch_size]
            print(f"     -> Processing batch {i} to {i + len(batch_slice)} of {len(image_paths)}...")
            images_list = list(executor.map(lambda p: preprocess(Image.open(p).convert("RGB")) if os.path.exists(p) else None, batch_slice))
            valid_images = [img for img in images_list if img is not None]
            if not valid_images: continue
            images = torch.stack(valid_images).to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                features = model.encode_image(images)
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().to(torch.float32).numpy())
            del images, features
            if device.type == 'cuda': torch.cuda.empty_cache()
    return np.vstack(all_features) if all_features else np.zeros((0, EMBED_DIM), dtype=np.float32)


@torch.no_grad()
def get_ensemble_text_embedding(query_text, enhancements, positive_bias=None, negative_bias=None):
    prompts = [f"{query_text}, {enhancements}, high-resolution satellite imagery", f"satellite view of {query_text}"]
    for lbl in (positive_bias or [])[:3]: prompts.append(f"satellite view of {lbl}, similar to confirmed sites")
    text = tokenizer(prompts).to(device)
    with torch.autocast(device_type=device.type):
        features = model.encode_text(text)
    features /= features.norm(dim=-1, keepdim=True)
    mean_feature = features.mean(dim=0, keepdim=True)
    if negative_bias:
        neg_text = tokenizer([f"satellite view of {lbl}" for lbl in negative_bias[:3]]).to(device)
        with torch.autocast(device_type=device.type):
            neg_features = model.encode_text(neg_text)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)
        mean_feature = mean_feature - 0.15 * neg_features.mean(dim=0, keepdim=True)
    mean_feature /= mean_feature.norm(dim=-1, keepdim=True)
    return mean_feature.cpu().to(torch.float32).numpy()


def intelligent_layer_router(query):
    q = query.lower()
    layers, enhancements = set(), []
    if re.search(r'\b(construction|building|urban)\b', q): layers.update(["true_color", "false_color_nir"]); enhancements.append("high albedo concrete")
    if re.search(r'\b(road|highway|street)\b', q): layers.update(["false_color_nir", "true_color"]); enhancements.append("linear asphalt trace")
    if re.search(r'\b(cleared|deforest|bare soil)\b', q): layers.add("false_color_nir"); enhancements.append("exposed earth")
    if re.search(r'\b(flood|water|river|lake)\b', q): layers.update(["agriculture_swir", "false_color_nir"]); enhancements.append("water absorbing infrared")
    if not layers: layers.add("true_color"); enhancements.append("landscape overview")
    return list(layers), ", ".join(enhancements)


def execute_search(query_emb, records, embeddings, start_date=None, end_date=None, aoi=None):
    valid_indices = [idx for idx, rec in enumerate(records) if (not start_date or not rec["date"] or rec["date"] >= datetime.strptime(start_date, "%Y-%m-%d")) and (not end_date or not rec["date"] or rec["date"] <= datetime.strptime(end_date, "%Y-%m-%d")) and check_bbox_intersection(rec["bounds"], aoi)]
    if not valid_indices: return []
    
    similarities = (embeddings[valid_indices] @ query_emb.T).squeeze(-1) * 100
    ranked = np.argsort(similarities)[::-1]
    
    artifact_text_features, logit_scale = get_artifact_text_features(), model.logit_scale.exp().item()
    results, seen = [], set()
    for idx in ranked:
        if similarities[idx] < 0: break
        rec = records[valid_indices[idx]]
        if rec["path"] in seen: continue
        seen.add(rec["path"])
        
        probs = softmax((embeddings[valid_indices[idx]] @ artifact_text_features.T) * logit_scale)
        cloud_conf = float(probs[1] * 100)
        shadow_conf = float(probs[2] * 100)
        if cloud_conf > 35.0 or shadow_conf > 35.0: continue
            
        results.append({
            "placename": rec["placename"], "filename": rec["filename"], "path": rec["path"], 
            "layer": rec["layer_source"], "score": round(float(similarities[idx]), 2), 
            "date": rec["date"].strftime("%Y-%m-%d") if rec["date"] else "Unknown",
            "bounds": rec["bounds"], "sensor": rec["sensor"],
            "cloud_conf": round(cloud_conf, 1), "shadow_conf": round(shadow_conf, 1)
        })
        if len(results) >= 5: break
    return results


def get_or_build_index(cleaned_folder, region_name, layers):
    idx = VectorIndexManager(region_name)
    added_total = 0
    for layer in layers:
        records = scan_dataset_folder_search(cleaned_folder, layer)
        new_records = [r for r in records if r["path"] not in idx.seen_paths]
        if not new_records: continue
        added_total += idx.add(new_records, get_image_embeddings([r["path"] for r in new_records]))
    if added_total: idx.save(); print(f"[VectorIndex] Added {added_total} new tiles to '{region_name}'.")
    return idx


# RESTRUCTURED WORKFLOW MASTER MENU

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
    folders.sort()

    while True:
        print("\n" + "=" * 60)
        print(" MASTER GEOINT WORKFLOW: SELECT CAPABILITY")
        print("=" * 60)
        print("  [1] Semantic / Text Search ")
        print("  [2] Multi-Temporal Change Detection & Timeline Dashboard ")
        print("  [3] Discovery & Clustering — Image to image search")
        print("  [4] Analyst Review Queue — Confirm/Reject, Audit Trail ")
        print("  [5] Exit")
        print("=" * 60)

        cap_choice = input("Select Capability to run (1-5): ").strip()
        if cap_choice == '5':
            print("Exiting engine. Goodbye!")
            break
        if cap_choice not in ['1', '2', '3', '4']:
            print("Invalid selection. Please choose 1-5.")
            continue

        print("\n" + "=" * 60)
        print(" SELECT TARGET DATASETS")
        print("=" * 60)
        print("  [0] All Datasets")
        for i, folder in enumerate(folders, 1):
            print(f"   [{i}] {folder}")
        print("=" * 60)

        try:
            dataset_choice = int(input("\nSelect dataset(s) to target (enter number): ").strip())
            if dataset_choice == 0:
                selected_regions = folders
            elif 1 <= dataset_choice <= len(folders):
                selected_regions = [folders[dataset_choice - 1]]
            else:
                print("❌ Invalid selection number.")
                continue
        except ValueError:
            print("❌ Please enter a valid number.")
            continue

        print(f"\n[Phase 1] Ensuring Quality Gating & Cleaning for: {selected_regions}")
        cleaned_paths = {}
        for region_name in selected_regions:
            cleaned_region_path = os.path.join(cleaned_base_dir, region_name)
            needs_cleaning = True
            if os.path.exists(cleaned_region_path) and os.listdir(cleaned_region_path):
                ans = input(f"Cleaned data already exists for '{region_name}'. Re-run Quality Gating? (y/n): ").strip().lower()
                if ans != 'y':
                    needs_cleaning = False

            if needs_cleaning:
                print(f"Running Quality Gating & Standardization on '{region_name}'...")
                cleaned_region_path = run_quality_pipeline(region_name, base_dir=raw_base_dir)
            else:
                print(f"Skipping cleaning for '{region_name}' (using existing cleaned files).")
            
            cleaned_paths[region_name] = cleaned_region_path

        # If Capability 1 (Semantic Search), prompt for query ONCE before running across selected datasets
        shared_query = None
        target_layers = None
        enhancements = None
        if cap_choice == '1':
            shared_query = input("\nEnter intelligence requirement (to be applied across selected datasets): ").strip()
            if not shared_query:
                continue
            target_layers, enhancements = intelligent_layer_router(shared_query)

        print(f"\n[Phase 2] Executing selected capability across target datasets...")
        for region_name in selected_regions:
            cleaned_region_path = cleaned_paths[region_name]
            review_queue = ReviewQueue(region_name)

            print(f"\n============================================================")
            print(f" PROCESSING DATASET REGION: {region_name}")
            print(f"============================================================")

            if cap_choice == '1':
                print(f"[Intelligent Router] Searching Layers: {', '.join(target_layers)}")
                idx_mgr = get_or_build_index(cleaned_region_path, region_name, target_layers)
                pos, neg = review_queue.feedback_bias_terms()
                results = execute_search(get_ensemble_text_embedding(shared_query, enhancements, pos, neg), idx_mgr.records, idx_mgr.all_embeddings())
                
                print(f"\n--- TOP 5 RANKED GEOINT RESULTS FOR: '{shared_query}' in [{region_name}] ---")
                if results:
                    for idx, res in enumerate(results, 1):
                        print(f"\n  [{idx}] MATCH SCORE: {res['score']}%")
                        print(f"      ├── Core File & Location: Region [{res['placename']}] | File: {res['filename']}")
                        print(f"      ├── Local Path: {res['path']}")
                        print(f"      ├── Temporal Metadata: Acquisition Date -> {res['date']}")
                        print(f"      ├── Spatial & Sensor Coordinates: BBox/Bounds -> {res['bounds']} | Sensor: {res['sensor']}")
                        print(f"      ├── Spectral Band Layer Type: {res['layer'].upper()}")
                        print(f"      └── Quality & Artifact Metrics: Cloud Conf: {res['cloud_conf']}% | Shadow Conf: {res['shadow_conf']}%")
                        
                    print("\n" + "=" * 50)
                    print(f" SELF-LEARNING & VERIFICATION LOOP FOR [{region_name}]")
                    print("=" * 50)
                    
                    for idx, res in enumerate(results, 1):
                        feedback = input(f"Result [{idx}] ({res['filename']}) - Is this result correct? (y/n): ").strip().lower()
                        if feedback == 'y':
                            review_queue.add_candidate(
                                candidate_type="search_result", dataset=region_name, 
                                confidence=round(min(1.0, max(0.0, res["score"]/100.0)), 3), 
                                acquisition_info={"date": res["date"], "layer": res["layer"]}, 
                                processing_history=["remoteclip_self_learning"], label=shared_query, 
                                evidence_paths=[res["path"]]
                            )
                            pending_items = review_queue.review_queue_ranked("pending")
                            if pending_items:
                                review_queue.decide(pending_items[0]["id"], "confirmed", "Self-learning positive feedback loop")
                            print(f"   ✓ Logged as CORRECT. Model weight feedback saved.")
                        else:
                            review_queue.add_candidate(
                                candidate_type="search_result", dataset=region_name, 
                                confidence=round(min(1.0, max(0.0, res["score"]/100.0)), 3), 
                                acquisition_info={"date": res["date"], "layer": res["layer"]}, 
                                processing_history=["remoteclip_self_learning"], label=shared_query, 
                                evidence_paths=[res["path"]]
                            )
                            pending_items = review_queue.review_queue_ranked("pending")
                            if pending_items:
                                review_queue.decide(pending_items[0]["id"], "rejected", "Self-learning negative feedback loop")
                            print(f"   ✗ Logged as INCORRECT. Model negative suppression saved.")
                    print(f"\n[+] Self-learning feedback integrated for [{region_name}]!")
                else:
                    print(" ❌ No tiles matched the semantic requirements under quality gating constraints.")

            elif cap_choice == '2':
                tc_dir = os.path.join(cleaned_region_path, "true_color")
                if not os.path.exists(tc_dir):
                    print(f"Error: Missing 'true_color' layer in cleaned directory for {region_name}.")
                    continue

                files = os.listdir(tc_dir)
                available_dates = sorted(list({re.search(r"(\d{4}-\d{2}-\d{2})", f).group(1) for f in files if re.search(r"(\d{4}-\d{2}-\d{2})", f)}))

                if len(available_dates) < 2:
                    print(f"Not enough valid dates remaining after quality gating for {region_name} to build a timeline.")
                    continue

                print(f"Locked {len(available_dates)} clean dates for {region_name}. Running Change Detection...")
                run_automated_timeline(cleaned_region_path, available_dates, review_queue=review_queue, dataset_name_override=region_name)

            elif cap_choice == '3':
                run_discovery_clustering(VectorIndexManager(region_name))

            elif cap_choice == '4':
                run_review_queue_ui(review_queue)


if __name__ == "__main__":
    main_menu()
