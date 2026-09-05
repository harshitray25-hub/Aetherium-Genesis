"""
Enterprise GEOINT Engine — Full Implementation
================================================
Covers all 6 required capabilities from the SIH-2026 problem statement:

  2.2.1  Semantic & Multimodal Retrieval        -> run_semantic_search / run_image_to_image_search
  2.2.2  Multi-Temporal Change Analysis          -> run_automated_timeline (+ earliest-date estimation)
  2.2.3  False-Alarm Suppression & Quality        -> MultiBandQualityEnhancementPipeline
  2.2.4  Discovery & Clustering                  -> ClusterEngine / run_discovery_clustering
  2.2.5  Analyst Workflow & Provenance            -> ReviewQueue
  2.2.6  Scale, Incremental Ingestion, Sovereignty-> VectorIndexManager (FAISS, on-disk, incremental)
                                                     + GeoTIFF/COG ingestion via rasterio (optional)

Everything runs fully offline once weights/checkpoints are staged locally
(hf_hub_download(..., local_files_only=True)), satisfying 2.2.7.
"""

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

print("Initializing Enterprise GEOINT Engine (Capabilities 1-6)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

print(f"Loading RemoteCLIP ({model_name}) weights offline...")
MODEL_PROVENANCE = {
    "name": "RemoteCLIP",
    "backbone": model_name,
    "source": "huggingface.co/chendelong/RemoteCLIP",
    "license": "Apache-2.0 (verify against upstream repo before competition submission)",
    "staged_offline": True,
}
try:
    ckpt_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", local_files_only=True)
except Exception:
    ckpt_path = f"checkpoints/RemoteCLIP-{model_name}.pt"

if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    MODEL_PROVENANCE["checkpoint_path"] = ckpt_path
    MODEL_PROVENANCE["checkpoint_sha256"] = None
    try:
        with open(ckpt_path, "rb") as fh:
            MODEL_PROVENANCE["checkpoint_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    except Exception:
        pass
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


# ==========================================
# GEO INGESTION (2.2.6 — GeoTIFF / COG support)
# ==========================================
def get_real_bounds_from_geotiff(path):
    """Extract true WGS84 bounds from a GeoTIFF/COG's embedded georeferencing.
    Falls back to None if rasterio isn't available or the file has no CRS."""
    if not RASTERIO_AVAILABLE or not path.lower().endswith((".tif", ".tiff")):
        return None
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                return None
            b = src.bounds
            if src.crs.to_epsg() != 4326:
                left, bottom, right, top = transform_bounds(src.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
            else:
                left, bottom, right, top = b.left, b.bottom, b.right, b.top
            return [left, bottom, right, top]
    except Exception:
        return None


def resolve_bounds(path, placename):
    """Prefer real georeferencing embedded in the file (GeoTIFF/COG); fall back
    to the organiser-provided bounds table for flat jpg/png tiles."""
    real = get_real_bounds_from_geotiff(path)
    if real is not None:
        return real
    return DATASET_BOUNDS.get(placename, None)


# ==========================================
# SPECTRAL INDEX CALCULATIONS
# ==========================================
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
        bottom_strip = gray[h - tb:, :]
        left_strip = gray[:, :lr]
        right_strip = gray[:, w - lr:]

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
            if img is None:
                return 0.0
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
        if img is None:
            return None
        dehazed = self.dark_channel_dehaze(img)
        pil_img = Image.fromarray(cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB))
        pil_img = ImageEnhance.Color(pil_img).enhance(1.15)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.10)
        enhanced_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        normalized = self.match_histogram(enhanced_bgr, master_template)
        return cv2.bilateralFilter(normalized, d=5, sigmaColor=30, sigmaSpace=30)

    def quality_confidence(self, image_path):
        """0-1 confidence that this frame is analytically usable — feeds
        provenance/confidence fields in the review queue (2.2.5)."""
        img = cv2.imread(image_path)
        if img is None:
            return 0.0
        is_cloudy, cloud_pct = self.check_cloud_rejection(img)
        has_edges, edge_pct = self.check_black_borders_rejection(img)
        penalty = min(1.0, (cloud_pct / 100.0) + (edge_pct / 200.0))
        return round(max(0.0, 1.0 - penalty), 3)


def run_quality_pipeline(region_name, base_dir="./satellite_datasets"):
    target_directory = os.path.join(base_dir, region_name)
    master_cleaned_root = "./satellite_datasets_cleaned"
    region_output_root = os.path.join(master_cleaned_root, region_name)
    os.makedirs(region_output_root, exist_ok=True)

    target_bands = ["true_color", "false_color_nir", "agriculture_swir"]
    engine = MultiBandQualityEnhancementPipeline()
    quality_manifest = {}

    for band_name in target_bands:
        band_dir = os.path.join(target_directory, band_name)
        if not os.path.exists(band_dir):
            continue

        image_files = sorted(glob.glob(os.path.join(band_dir, "*.jpg")) + glob.glob(os.path.join(band_dir, "*.png")) + glob.glob(os.path.join(band_dir, "*.tif")))
        if not image_files:
            continue

        print(f"\n--- Processing Band: [{band_name.upper()}] ---")
        output_processed_dir = os.path.join(region_output_root, band_name)
        os.makedirs(output_processed_dir, exist_ok=True)

        valid_images = []
        for path in image_files:
            filename = os.path.basename(path)
            rejected, reason = engine.validate_frame(path)
            if rejected:
                print(f"  [\u274c REJECTED] {filename} -> {reason}")
                quality_manifest[filename] = {"status": "rejected", "reason": reason}
            else:
                print(f"  [\u2713 PASSED]   {filename}")
                valid_images.append(path)

        if not valid_images:
            continue

        print("Evaluating surviving images to find the master reference standard...")
        best_image_path = max(valid_images, key=lambda p: engine.evaluate_image_clarity(p))
        master_template = cv2.imread(best_image_path)
        print(f"[*] Master Reference Standard: {os.path.basename(best_image_path)}")

        for path in valid_images:
            filename = os.path.basename(path)
            cleaned_img = engine.process_and_standardize_image(path, master_template)
            confidence = engine.quality_confidence(path)
            quality_manifest[filename] = {
                "status": "passed",
                "confidence": confidence,
                "reference_used": os.path.basename(best_image_path),
                "band": band_name,
            }
            if cleaned_img is not None:
                cv2.imwrite(os.path.join(output_processed_dir, filename), cleaned_img)

    with open(os.path.join(region_output_root, "quality_manifest.json"), "w") as f:
        json.dump(quality_manifest, f, indent=2)

    print(f"\n[+] Quality & Standardization complete. Cleaned data saved to: '{region_output_root}'")
    return region_output_root


# ==========================================
# CAPABILITY 2: CHANGE DETECTION (+ earliest-date estimation)
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
                if extent < active_min_extent:
                    continue
                kernel = np.ones((3, 3), np.uint8)
                edges = cv2.Canny(cv2.dilate(merged_mask[y_int:y_int + h, x_int:x_int + w], kernel, iterations=1), 50, 150, apertureSize=3)
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=25, maxLineGap=10)
                if lines is not None and len(lines) > 0:
                    if aspect_ratio >= MIN_ROAD_ASPECT_RATIO and compactness <= MAX_ROAD_COMPACTNESS:
                        final_category, final_color = "Road Development", (0, 165, 255)
                    else:
                        final_category, final_color = "Construction", (255, 0, 255)

            approx_contour = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            events_list.append({
                "type": final_category, "center": (cx, cy), "contour": approx_contour,
                "area": int(area), "color": final_color, "is_global": False,
                # rough confidence: bigger/solid/interior blobs score higher
                "confidence": round(float(min(1.0, 0.4 + 0.3 * min(solidity, 1.0) + 0.3 * min(area / 5000.0, 1.0))), 3),
            })
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


def compute_change_events(t1_rgb, t2_rgb, t1_nir, t2_nir):
    """Pure change-detection core, factored out so it can be reused both for
    consecutive-pair timelines and for earliest-date backtracking."""
    img_h, img_w = t1_rgb.shape[:2]
    img_shape = t1_rgb.shape
    events = []

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g1 = clahe.apply(cv2.cvtColor(t1_rgb, cv2.COLOR_BGR2GRAY))
    g2 = clahe.apply(cv2.cvtColor(t2_rgb, cv2.COLOR_BGR2GRAY))

    _, diff = ssim(cv2.bilateralFilter(g1, 5, 50, 50), cv2.bilateralFilter(g2, 5, 50, 50), win_size=11, data_range=255, full=True)
    _, structure_change = cv2.threshold(cv2.bitwise_not((diff * 255).astype(np.uint8)), int((1 - SSIM_THRESH) * 255), 255, cv2.THRESH_BINARY)
    structure_change = cv2.morphologyEx(cv2.morphologyEx(structure_change, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

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

    return events, img_h, img_w


def estimate_earliest_change_dates(target_folder, available_dates, final_events, tolerance_px=40):
    """2.2.2: 'estimate the earliest available observation at which the change
    is supported by usable imagery'. For each feature detected between the
    last two dates, walk backwards through the full date sequence comparing
    each earlier frame against the most recent frame; the earliest date whose
    local patch already shows the same structural change becomes the
    earliest-support date. This uses the FULL history, not just one pair."""
    if len(available_dates) < 3 or not final_events:
        return {e["center"]: available_dates[-2] if len(available_dates) >= 2 else None for e in final_events}

    tc_dir = os.path.join(target_folder, "true_color")
    fc_dir = os.path.join(target_folder, "false_color_nir")
    latest_date = available_dates[-1]
    latest_rgb = load_layer_cd(latest_date, "true_color", tc_dir)
    if latest_rgb is None:
        return {}

    earliest_map = {}
    for e in final_events:
        if e["is_global"] or e["contour"] is None:
            continue
        cx, cy = e["center"]
        x0, x1 = max(0, cx - tolerance_px), cx + tolerance_px
        y0, y1 = max(0, cy - tolerance_px), cy + tolerance_px
        earliest_supporting = available_dates[-2]  # default: only supported by the last pair

        for candidate_date in available_dates[:-1]:
            candidate_rgb = load_layer_cd(candidate_date, "true_color", tc_dir)
            if candidate_rgb is None:
                continue
            candidate_rgb = cv2.resize(candidate_rgb, (latest_rgb.shape[1], latest_rgb.shape[0])) if candidate_rgb.shape != latest_rgb.shape else candidate_rgb

            patch_latest = latest_rgb[y0:y1, x0:x1]
            patch_candidate = candidate_rgb[y0:y1, x0:x1]
            if patch_latest.size == 0 or patch_candidate.size == 0 or patch_latest.shape != patch_candidate.shape:
                continue

            g_latest = cv2.cvtColor(patch_latest, cv2.COLOR_BGR2GRAY)
            g_candidate = cv2.cvtColor(patch_candidate, cv2.COLOR_BGR2GRAY)
            try:
                score = ssim(g_latest, g_candidate, data_range=255)
            except Exception:
                continue

            # Low similarity to the "before" state at this date means the change
            # already existed by this date -> keep pushing the earliest date back.
            if score < SSIM_THRESH:
                earliest_supporting = candidate_date
            else:
                # Found a date where it still looks like "before" -> stop, the
                # change did not yet exist at this date.
                break

        earliest_map[(cx, cy)] = earliest_supporting
    return earliest_map


def process_timeframe(d1, d2, layers, target_folder, visuals_dir, dataset_bounds, all_dates=None):
    t1_imgs, t2_imgs = {}, {}
    for layer in layers:
        l_folder = os.path.join(target_folder, layer)
        img1 = load_layer_cd(d1, layer, l_folder)
        img2 = load_layer_cd(d2, layer, l_folder)
        if img1 is None or img2 is None:
            return None
        t1_imgs[layer], t2_imgs[layer] = align_shapes(img1, img2)

    t1_rgb, t2_rgb = t1_imgs["true_color"], t2_imgs["true_color"]
    t1_nir, t2_nir = t1_imgs["false_color_nir"], t2_imgs["false_color_nir"]

    events, img_h, img_w = compute_change_events(t1_rgb, t2_rgb, t1_nir, t2_nir)

    earliest_dates = {}
    if all_dates and all_dates[-1] == d2:
        earliest_dates = estimate_earliest_change_dates(target_folder, all_dates, events)

    payload = {"timeframe": f"{d1}_to_{d2}", "features": []}

    visual_proof = generate_dashboard_image(t2_rgb, events, d1, d2)
    cv2.imwrite(os.path.join(visuals_dir, f"Polygon_Dashboard_{d1}_to_{d2}.jpg"), visual_proof)

    for e in events:
        if e["is_global"] or e["contour"] is None:
            continue
        coord_list = []
        for point in e["contour"]:
            px, py = int(point[0][0]), int(point[0][1])
            coord_list.append(pixel_to_latlon(px, py, img_w, img_h, dataset_bounds) if dataset_bounds else [float(px), float(py)])

        if len(coord_list) >= 3:
            if coord_list[0] != coord_list[-1]:
                coord_list.append(coord_list[0])
            earliest_date = earliest_dates.get(e["center"], d1)
            payload["features"].append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coord_list]},
                "properties": {
                    "event_type": str(e["type"]),
                    "area_pixels": int(e["area"]),
                    "timeframe_start": str(d1),
                    "timeframe_end": str(d2),
                    "earliest_supported_observation": str(earliest_date),
                    "confidence": e.get("confidence", 0.5),
                    "processing_history": [
                        "quality_gate_v1", "dark_channel_dehaze", "histogram_match",
                        "ecc_translation_align", "ssim_structure_diff", "ndvi_ndwi_index",
                        "contour_extraction", "earliest_date_backtrack",
                    ],
                }
            })

    del t1_imgs, t2_imgs, t1_rgb, t2_rgb, t1_nir, t2_nir
    gc.collect()
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

    safe_workers = max(1, min(4, os.cpu_count() or 4))
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=safe_workers) as executor:
        future_to_pair = {
            executor.submit(process_timeframe, p[0], p[1], layers, target_folder, visuals_dir, dataset_bounds, available_dates): p
            for p in pairs
        }
        for future in concurrent.futures.as_completed(future_to_pair):
            pair = future_to_pair[future]
            completed += 1
            print(f"\r[ {completed} / {len(pairs)} ] Processed: {pair[0]} \u2794 {pair[1]}", end="", flush=True)
            try:
                result = future.result()
                if result:
                    master_geojson["features"].extend(result["features"])
            except Exception as exc:
                print(f"\n[!] Error processing {pair[0]} \u2794 {pair[1]}: {exc}")

    with open(os.path.join(visuals_dir, "timeline_analysis_data.geojson"), 'w') as f:
        json.dump(master_geojson, f, indent=4)

    # 2.2.5: push every detected change into the analyst review queue with
    # full provenance (source scene dates, confidence, processing history).
    if review_queue is not None:
        for feat in master_geojson["features"]:
            props = feat["properties"]
            review_queue.add_candidate(
                candidate_type="change_event",
                dataset=dataset_name,
                geometry=feat["geometry"],
                confidence=props.get("confidence", 0.5),
                acquisition_info={
                    "timeframe_start": props["timeframe_start"],
                    "timeframe_end": props["timeframe_end"],
                    "earliest_supported_observation": props["earliest_supported_observation"],
                    "sensor": "organiser-supplied multi-band tiles (true_color/false_color_nir/agriculture_swir)",
                },
                processing_history=props.get("processing_history", []),
                label=props["event_type"],
                evidence_paths=[os.path.join(visuals_dir, f"Polygon_Dashboard_{props['timeframe_start']}_to_{props['timeframe_end']}.jpg")],
            )

    print("\n" + "=" * 70)
    print(f"Pipeline Complete in {round(time.time() - start_time, 2)}s.")
    print(f"Visual Dashboards & GeoJSON Vectors saved in: \n{visuals_dir}")
    print("=" * 70)


# ==========================================
# CAPABILITY 2.2.6: VECTOR INDEX MANAGER (FAISS, incremental, on-disk)
# ==========================================
class VectorIndexManager:
    """Persistent, incrementally-updatable vector index per dataset region.
    Avoids full-archive re-embedding on every new acquisition and satisfies
    the on-premises / no-cloud constraint (pure local FAISS index on disk)."""

    def __init__(self, region_name, dim=EMBED_DIM):
        self.region_name = region_name
        self.dim = dim
        self.index_path = os.path.join(INDEX_ROOT, f"{region_name}.faiss")
        self.meta_path = os.path.join(INDEX_ROOT, f"{region_name}_meta.pkl")
        self.records = []  # list of dicts, position == faiss vector id
        self.seen_paths = set()
        self._load_or_init()

    def _load_or_init(self):
        if FAISS_AVAILABLE and os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.records = pickle.load(f)
            self.seen_paths = {r["path"] for r in self.records}
            print(f"[VectorIndex] Loaded existing index for '{self.region_name}' ({len(self.records)} vectors).")
        elif FAISS_AVAILABLE:
            base = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIDMap2(base)
            print(f"[VectorIndex] Created new FAISS index for '{self.region_name}'.")
        else:
            self.index = None
            print("[VectorIndex] FAISS not available — falling back to in-memory numpy search (no persistence).")
            self._np_embeddings = np.zeros((0, self.dim), dtype=np.float32)

    def add(self, new_records, embeddings):
        """Incremental add: only embeds/ingests records not already indexed."""
        fresh_idx = [i for i, r in enumerate(new_records) if r["path"] not in self.seen_paths]
        if not fresh_idx:
            return 0
        fresh_records = [new_records[i] for i in fresh_idx]
        fresh_embs = embeddings[fresh_idx].astype(np.float32)

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
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.records, f)
        print(f"[VectorIndex] Saved '{self.region_name}': {len(self.records)} vectors -> {self.index_path}")

    def search(self, query_emb, top_k=50):
        query_emb = query_emb.astype(np.float32)
        if FAISS_AVAILABLE and self.index is not None:
            if self.index.ntotal == 0:
                return [], np.array([])
            scores, ids = self.index.search(query_emb, min(top_k, self.index.ntotal))
            ids, scores = ids[0], scores[0]
            valid = ids >= 0
            return [self.records[i] for i in ids[valid]], scores[valid]
        else:
            if self._np_embeddings.shape[0] == 0:
                return [], np.array([])
            sims = (self._np_embeddings @ query_emb.T).squeeze(-1)
            order = np.argsort(sims)[::-1][:top_k]
            return [self.records[i] for i in order], sims[order]

    def all_embeddings(self):
        """Reconstruct all vectors — used by the clustering engine (2.2.4)."""
        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            return np.vstack([self.index.reconstruct(i) for i in range(self.index.ntotal)])
        elif not FAISS_AVAILABLE:
            return self._np_embeddings
        return np.zeros((0, self.dim), dtype=np.float32)


# ==========================================
# CAPABILITY 4 (2.2.4): DISCOVERY & CLUSTERING
# ==========================================
def _kmeans_cluster(embeddings, k):
    if FAISS_AVAILABLE:
        km = faiss.Kmeans(embeddings.shape[1], k, niter=25, seed=42, spherical=True)
        km.train(embeddings.astype(np.float32))
        _, labels = km.index.search(embeddings.astype(np.float32), 1)
        return labels.reshape(-1)
    # Minimal numpy KMeans fallback (cosine-normalized vectors -> spherical kmeans)
    rng = np.random.default_rng(42)
    centroids = embeddings[rng.choice(len(embeddings), size=k, replace=False)].copy()
    labels = np.zeros(len(embeddings), dtype=int)
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
    """2.2.4: unsupervised grouping of similar sites across the whole indexed
    area. Returns {cluster_id: [record, ...]}."""
    embeddings = index_manager.all_embeddings()
    if embeddings.shape[0] < n_clusters:
        n_clusters = max(1, embeddings.shape[0])
    if embeddings.shape[0] == 0:
        return {}
    labels = _kmeans_cluster(embeddings, n_clusters)
    groups = {}
    for rec, lbl in zip(index_manager.records, labels):
        groups.setdefault(int(lbl), []).append(rec)
    return groups


def discover_similar_to_seed(index_manager, seed_path, n_clusters=8, top_k=15):
    """Given one location of interest, find the other members of its cluster
    (same visual/semantic 'type') without the analyst writing a new query."""
    groups = cluster_dataset(index_manager, n_clusters=n_clusters)
    for cluster_id, members in groups.items():
        for m in members:
            if os.path.abspath(m["path"]) == os.path.abspath(seed_path):
                others = [m2 for m2 in members if os.path.abspath(m2["path"]) != os.path.abspath(seed_path)]
                return cluster_id, others[:top_k]
    return None, []


def run_discovery_clustering(index_manager):
    print("\n--- DISCOVERY & CLUSTERING (2.2.4) ---")
    print("[a] Cluster the entire indexed region into similar-site groups")
    print("[b] Give a seed tile -> find other sites like it across the region")
    sub = input("Select (a/b): ").strip().lower()

    if sub == 'a':
        try:
            k = int(input("Number of clusters (e.g. 8): ").strip() or "8")
        except ValueError:
            k = 8
        groups = cluster_dataset(index_manager, n_clusters=k)
        for cid, members in sorted(groups.items()):
            print(f"\nCluster {cid} ({len(members)} tiles):")
            for m in members[:8]:
                print(f"    - {m['filename']} ({m.get('date')})")
            if len(members) > 8:
                print(f"    ... +{len(members) - 8} more")
    elif sub == 'b':
        seed = input("Filename or path of the seed tile of interest: ").strip().strip('"').strip("'")
        match = next((r for r in index_manager.records if r["filename"] == os.path.basename(seed) or os.path.abspath(r["path"]) == os.path.abspath(seed)), None)
        if not match:
            print("  \u274c Seed tile not found in the index. Run a search/ingest first.")
            return
        cid, others = discover_similar_to_seed(index_manager, match["path"])
        if cid is None:
            print("  \u274c Could not resolve a cluster for that tile.")
            return
        print(f"\nSeed tile belongs to cluster {cid}. Other sites with comparable characteristics:")
        for o in others:
            print(f"    - {o['filename']} ({o.get('date')}) @ {o.get('placename')}")
    else:
        print("Invalid selection.")


# ==========================================
# CAPABILITY 5 (2.2.5): ANALYST REVIEW QUEUE & PROVENANCE
# ==========================================
class ReviewQueue:
    """Ranked review queue with before/after evidence, provenance and an
    audit trail of analyst decisions. Confirmed/rejected feedback nudges
    future semantic-search ranking (simple relevance feedback)."""

    def __init__(self, region_name):
        self.region_name = region_name
        self.path = os.path.join(REVIEW_ROOT, f"{region_name}_review_queue.json")
        self.audit_path = os.path.join(REVIEW_ROOT, f"{region_name}_audit_trail.jsonl")
        self.items = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.items, f, indent=2)

    def _audit(self, entry):
        with open(self.audit_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def add_candidate(self, candidate_type, dataset, confidence, acquisition_info,
                       processing_history, label="", geometry=None, evidence_paths=None,
                       embedding=None):
        cand_id = str(uuid.uuid4())[:8]
        self.items[cand_id] = {
            "id": cand_id,
            "type": candidate_type,       # "change_event" | "search_result"
            "dataset": dataset,
            "label": label,
            "confidence": confidence,
            "geometry": geometry,
            "evidence_paths": evidence_paths or [],
            "acquisition_info": acquisition_info,
            "processing_history": processing_history,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "decision_at": None,
            "analyst_notes": None,
        }
        self._save()
        return cand_id

    def review_queue_ranked(self, status="pending"):
        items = [v for v in self.items.values() if v["status"] == status]
        return sorted(items, key=lambda x: x["confidence"], reverse=True)

    def decide(self, cand_id, decision, notes=None):
        if cand_id not in self.items:
            print("  \u274c Unknown candidate id.")
            return False
        assert decision in ("confirmed", "rejected")
        self.items[cand_id]["status"] = decision
        self.items[cand_id]["decision_at"] = datetime.now(timezone.utc).isoformat()
        self.items[cand_id]["analyst_notes"] = notes
        self._save()
        self._audit({
            "candidate_id": cand_id, "decision": decision, "notes": notes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": self.items[cand_id]["dataset"], "label": self.items[cand_id]["label"],
        })
        return True

    def feedback_bias_terms(self):
        """Turn confirm/reject history into extra positive/negative text
        prompts that bias future semantic search ranking (2.2.5's
        'use feedback for subsequent reranking')."""
        confirmed_labels = [v["label"] for v in self.items.values() if v["status"] == "confirmed" and v["label"]]
        rejected_labels = [v["label"] for v in self.items.values() if v["status"] == "rejected" and v["label"]]
        return confirmed_labels, rejected_labels


def run_review_queue_ui(review_queue):
    print("\n--- ANALYST REVIEW QUEUE (2.2.5) ---")
    pending = review_queue.review_queue_ranked("pending")
    if not pending:
        print("  Queue is empty — nothing pending review.")
        return
    for item in pending[:20]:
        print(f"\n[{item['id']}] {item['label']}  |  confidence={item['confidence']}  |  dataset={item['dataset']}")
        print(f"    acquisition: {item['acquisition_info']}")
        print(f"    evidence:    {item['evidence_paths']}")
        print(f"    processing:  {', '.join(item['processing_history'])}")

    choice = input("\nEnter candidate id to confirm/reject (blank to skip): ").strip()
    if not choice:
        return
    decision = input("Decision (confirm/reject): ").strip().lower()
    notes = input("Analyst notes (optional): ").strip() or None
    if decision.startswith("c"):
        review_queue.decide(choice, "confirmed", notes)
        print("  \u2713 Marked confirmed and logged to audit trail.")
    elif decision.startswith("r"):
        review_queue.decide(choice, "rejected", notes)
        print("  \u2713 Marked rejected and logged to audit trail.")
    else:
        print("  Invalid decision, no change made.")


# ==========================================
# CAPABILITY 1: SEMANTIC & VECTOR SEARCH (now index-backed, 2.2.1 + 2.2.6)
# ==========================================
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
        if not is_target_layer:
            continue
        placename = os.path.basename(base_folder)

        for f in files:
            if f.lower().endswith(valid_extensions):
                full_path = os.path.join(root, f)
                image_records.append({
                    "path": full_path,
                    "filename": f,
                    "placename": placename,
                    "bounds": resolve_bounds(full_path, placename),
                    "date": parse_date_from_filename(f),
                    "layer_source": layer_type
                })
    return sorted(image_records, key=lambda x: x["filename"])


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
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(0, len(image_paths), batch_size):
            images_list = list(executor.map(load_and_preprocess, image_paths[i:i + batch_size]))
            valid_images = [img for img in images_list if img is not None]
            if not valid_images:
                continue

            images = torch.stack(valid_images).to(device)
            with torch.autocast(device_type=device.type):
                features = model.encode_image(images)
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().to(torch.float32).numpy())
            del images, features
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    return np.vstack(all_features) if all_features else np.zeros((0, EMBED_DIM), dtype=np.float32)


@torch.no_grad()
def get_ensemble_text_embedding(query_text, enhancements, positive_bias=None, negative_bias=None):
    prompts = [f"{query_text}, {enhancements}, high-resolution satellite imagery", f"satellite view of {query_text}"]
    # 2.2.5 feedback loop: fold confirmed/rejected labels into the query ensemble
    for lbl in (positive_bias or [])[:3]:
        prompts.append(f"satellite view of {lbl}, similar to previously confirmed sites")
    text = tokenizer(prompts).to(device)
    with torch.autocast(device_type=device.type):
        features = model.encode_text(text)
    features /= features.norm(dim=-1, keepdim=True)
    mean_feature = features.mean(dim=0, keepdim=True)

    if negative_bias:
        neg_prompts = [f"satellite view of {lbl}" for lbl in negative_bias[:3]]
        neg_text = tokenizer(neg_prompts).to(device)
        with torch.autocast(device_type=device.type):
            neg_features = model.encode_text(neg_text)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)
        mean_feature = mean_feature - 0.15 * neg_features.mean(dim=0, keepdim=True)

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
        if start_date and rec["date"] and rec["date"] < datetime.strptime(start_date, "%Y-%m-%d"):
            continue
        if end_date and rec["date"] and rec["date"] > datetime.strptime(end_date, "%Y-%m-%d"):
            continue
        if aoi and not check_bbox_intersection(rec["bounds"], aoi):
            continue
        valid_indices.append(idx)

    if not valid_indices:
        return []
    f_records = [records[i] for i in valid_indices]
    f_embeddings = embeddings[valid_indices]

    similarities = (f_embeddings @ query_emb.T).squeeze(-1) * 100
    ranked_indices = np.argsort(similarities)[::-1]

    artifact_text_features = get_artifact_text_features()
    logit_scale = model.logit_scale.exp().item()

    results, seen_paths = [], set()
    for idx in ranked_indices:
        if similarities[idx] < 0:
            break
        rec = f_records[idx]
        if rec["path"] in seen_paths:
            continue
        seen_paths.add(rec["path"])

        logits = (f_embeddings[idx] @ artifact_text_features.T) * logit_scale
        probs = softmax(logits)
        if probs[1] * 100 > 35.0 or probs[2] * 100 > 35.0:
            continue

        results.append({
            "placename": rec["placename"], "filename": rec["filename"], "path": rec["path"],
            "layer": rec["layer_source"], "score": round(float(similarities[idx]), 2),
            "date": rec["date"].strftime("%Y-%m-%d") if rec["date"] else None,
            "bounds": rec.get("bounds"),
        })
        if len(results) >= 15:
            break
    return results


def get_or_build_index(cleaned_folder, region_name, layers):
    """2.2.6: build once, then incrementally add only unseen tiles on
    subsequent runs instead of a full re-embed / re-index."""
    idx = VectorIndexManager(region_name)
    added_total = 0
    for layer in layers:
        records = scan_dataset_folder_search(cleaned_folder, layer)
        new_records = [r for r in records if r["path"] not in idx.seen_paths]
        if not new_records:
            continue
        embs = get_image_embeddings([r["path"] for r in new_records])
        added = idx.add(new_records, embs)
        added_total += added
    if added_total:
        idx.save()
        print(f"[VectorIndex] Incrementally added {added_total} new tiles to '{region_name}' index.")
    else:
        print(f"[VectorIndex] Index for '{region_name}' already up to date ({len(idx.records)} tiles).")
    return idx


def run_semantic_search(cleaned_folder, region_name, review_queue=None):
    user_query = input("\nEnter intelligence requirement (e.g., 'newly built concrete structures'): ").strip()
    if not user_query:
        return

    target_layers, enhancements = intelligent_layer_router(user_query)
    print(f"\n[Intelligent Router] Searching Layers: {', '.join(target_layers)}")

    index_manager = get_or_build_index(cleaned_folder, region_name, target_layers)
    if not index_manager.records:
        print(" \u274c No tiles found in the cleaned dataset for those layers.")
        return

    pos_bias, neg_bias = (review_queue.feedback_bias_terms() if review_queue else ([], []))
    text_emb = get_ensemble_text_embedding(user_query, enhancements, positive_bias=pos_bias, negative_bias=neg_bias)

    all_records = index_manager.records
    all_embeddings = index_manager.all_embeddings()
    results = execute_search(text_emb, all_records, all_embeddings)

    print(f"\n--- RANKED GEOINT RESULTS FOR: '{user_query}' ---")
    if results:
        for idx, res in enumerate(results, 1):
            print(f"  {idx}. [Score: {res['score']}%] | Band: {res['layer']} | Date: {res['date']} | Tile: {res['filename']}")
        if review_queue is not None:
            push = input("\nPush top results to the analyst review queue for confirm/reject? (y/n): ").strip().lower()
            if push == 'y':
                for res in results:
                    review_queue.add_candidate(
                        candidate_type="search_result",
                        dataset=region_name,
                        confidence=round(min(1.0, max(0.0, res["score"] / 100.0)), 3),
                        acquisition_info={"date": res["date"], "layer": res["layer"], "sensor": res["layer"]},
                        processing_history=["quality_gate_v1", "remoteclip_embedding", "cosine_similarity_rank"],
                        label=user_query,
                        evidence_paths=[res["path"]],
                    )
                print("  \u2713 Pushed to review queue.")
    else:
        print("  \u274c No tiles matched the semantic requirements.")


def run_image_to_image_search(cleaned_folder, region_name):
    img_path = input("\nEnter filename or relative path of reference tile: ").strip().strip('"').strip("'")
    if not img_path:
        return

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
        print("  \u274c Error: Reference image tile could not be found locally across dataset folders.")
        return

    print(f"\n[Cross-Spectral Router] Found reference in layer band: '{ref_layer}'")
    index_manager = get_or_build_index(cleaned_folder, f"{region_name}_{ref_layer}", [ref_layer])
    ref_emb = get_image_embeddings([found_ref])

    results = execute_search(ref_emb, index_manager.records, index_manager.all_embeddings())
    results = [r for r in results if os.path.abspath(r["path"]) != os.path.abspath(found_ref)]

    print(f"\n--- RANKED CROSS-SPECTRAL MATCHES ---")
    if results:
        for idx, res in enumerate(results[:15], 1):
            print(f"  {idx}. [Score: {res['score']}%] | Band: {res['layer']} | Date: {res['date']} | Tile: {res['filename']}")
    else:
        print("  \u274c No matching cross-spectral neighbors found.")
    return index_manager


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
        if folder_choice < 0 or folder_choice >= len(folders):
            return
    except ValueError:
        return

    region_name = folders[folder_choice]
    cleaned_region_path = os.path.join(cleaned_base_dir, region_name)
    review_queue = ReviewQueue(region_name)

    needs_cleaning = True
    if os.path.exists(cleaned_region_path) and os.listdir(cleaned_region_path):
        ans = input(f"\nCleaned data already exists for '{region_name}'. Re-run Quality Gating & Cleaning? (y/n): ").strip().lower()
        if ans != 'y':
            needs_cleaning = False

    if needs_cleaning:
        print(f"\n[Phase 1] Running Capability 3 Quality Gating & Standardization on '{region_name}'...")
        cleaned_region_path = run_quality_pipeline(region_name, base_dir=raw_base_dir)

    # Pre-build/refresh the persistent vector index up front (2.2.6)
    default_index = get_or_build_index(cleaned_region_path, region_name, ["true_color", "false_color_nir", "agriculture_swir"])

    while True:
        print("\n" + "=" * 60)
        print(f" ACTIVE REGION: {region_name} (Using Cleaned Data)")
        print("=" * 60)
        print("  [1] Run GIS Change Detection & Timeline Dashboard (Cap 2)")
        print("  [2] Plain English / Semantic Text Search (Cap 1)")
        print("  [3] Image-to-Image Cross-Spectral Search (Cap 1/4)")
        print("  [4] Discovery & Clustering — find similar sites (Cap 4)")
        print("  [5] Analyst Review Queue — confirm/reject, audit trail (Cap 5)")
        print("  [6] Exit")
        print("=" * 60)

        choice = input("Select feature to run (1-6): ").strip()
        if choice == '6':
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
            run_automated_timeline(cleaned_region_path, available_dates, review_queue=review_queue, dataset_name_override=region_name)

        elif choice == '2':
            print(f"\n[Phase 2] Running Semantic Vector Search on clean data...")
            run_semantic_search(cleaned_region_path, region_name, review_queue=review_queue)

        elif choice == '3':
            print(f"\n[Phase 2] Running Image-to-Image Cross-Spectral Search on clean data...")
            run_image_to_image_search(cleaned_region_path, region_name)

        elif choice == '4':
            run_discovery_clustering(default_index)

        elif choice == '5':
            run_review_queue_ui(review_queue)

        else:
            print("Invalid selection. Please choose 1-6.")


if __name__ == "__main__":
    main_menu()