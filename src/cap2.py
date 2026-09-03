import os
import cv2
import numpy as np
import glob
import re
import time
import json
import gc
import concurrent.futures
from skimage.metrics import structural_similarity as ssim

# ==========================================
# ADVANCED TUNING PARAMETERS
# ==========================================
NDWI_WATER_THRESH = 0.10        
NDVI_VEG_THRESH = 0.25          
NDVI_DROP_MIN = 0.12            
MIN_CHANGE_AREA_PX = 400        
SSIM_THRESH = 0.55              
GLOBAL_SEASONAL_LIMIT = 0.09    
EDGE_BUFFER = 15                

# GEOMETRIC PROFILING LIMITS
MIN_ROAD_ASPECT_RATIO = 3.0     
MAX_ROAD_COMPACTNESS = 0.20     
FARM_SOLIDITY_THRESH = 0.50     
MIN_CONSTRUCTION_EXTENT = 0.25  

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

def load_layer(date_prefix, layer_name, folder_path):
    pattern = os.path.join(folder_path, f"{date_prefix}*_{layer_name}.jpg")
    files = glob.glob(pattern)
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

def get_nodata_mask(rgb_img):
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    _, nodata = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5), np.uint8)
    return cv2.dilate(nodata, kernel, iterations=2)

def create_cloud_mask(rgb_img):
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    
    _, bright_mask = cv2.threshold(v, 160, 255, cv2.THRESH_BINARY)
    _, low_sat_mask = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)
    base_cloud = cv2.bitwise_and(bright_mask, low_sat_mask)
    
    open_kernel = np.ones((7, 7), np.uint8)
    cloud_cores = cv2.morphologyEx(base_cloud, cv2.MORPH_OPEN, open_kernel)
    dilate_kernel = np.ones((35, 35), np.uint8)
    return cv2.dilate(cloud_cores, dilate_kernel, iterations=1)

def create_shadow_mask(rgb_img):
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    _, shadow_mask = cv2.threshold(v, 35, 255, cv2.THRESH_BINARY_INV)
    
    clean_kernel = np.ones((5, 5), np.uint8)
    shadow_cores = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, clean_kernel)
    dilate_kernel = np.ones((15, 15), np.uint8)
    return cv2.dilate(shadow_cores, dilate_kernel, iterations=1)

def get_structural_change_mask(t1_rgb, t2_rgb):
    g1 = cv2.cvtColor(t1_rgb, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(t2_rgb, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g1_eq = clahe.apply(g1)
    g2_eq = clahe.apply(g2)
    
    _, diff = ssim(cv2.bilateralFilter(g1_eq, 5, 50, 50), cv2.bilateralFilter(g2_eq, 5, 50, 50), win_size=11, data_range=255, full=True)
    inv_diff = cv2.bitwise_not((diff * 255).astype(np.uint8))
    _, thresh = cv2.threshold(inv_diff, int((1 - SSIM_THRESH) * 255), 255, cv2.THRESH_BINARY)
    
    clean_kernel = np.ones((3,3), np.uint8)
    scrubbed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel)
    merge_kernel = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(scrubbed, cv2.MORPH_CLOSE, merge_kernel)

def calculate_ndvi(false_color_img):
    img_float = false_color_img.astype(np.float32)
    nir = img_float[:, :, 2]
    red = img_float[:, :, 1]
    denominator = (nir + red)
    denominator[denominator == 0] = 1e-5
    return (nir - red) / denominator

def calculate_ndwi(true_color_img, false_color_img):
    green = true_color_img.astype(np.float32)[:, :, 1]
    nir = false_color_img.astype(np.float32)[:, :, 2]
    denominator = (green + nir)
    denominator[denominator == 0] = 1e-5
    return (green - nir) / denominator

def detect_linear_features(mask_roi):
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(mask_roi, kernel, iterations=1)
    edges = cv2.Canny(dilated, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=25, maxLineGap=10)
    return lines is not None and len(lines) > 0

def extract_change_clusters(mask, category_name, color, events_list, img_shape, check_linear=False, skip=False, strict_mode=False):
    if skip:
        events_list.append({
            "type": f"Global Shift ({category_name})",
            "center": (0, 0),
            "contour": None,
            "area": 0,
            "color": (128, 128, 128),
            "is_global": True
        })
        return events_list

    # Dynamic Gating parameters based on atmospheric/seasonal stability
    active_min_area = MIN_CHANGE_AREA_PX * 2 if strict_mode else MIN_CHANGE_AREA_PX
    active_min_extent = 0.40 if strict_mode else MIN_CONSTRUCTION_EXTENT

    if "Clearance" in category_name or "Growth" in category_name:
        k_size = 9  
    else:
        k_size = 15 

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
            (x_rect, y_rect), (w_rect, h_rect), angle = rect
            aspect_ratio = max(w_rect, h_rect) / (min(w_rect, h_rect) + 1e-5)
            
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            bounding_area = w * h
            extent = float(area) / bounding_area if bounding_area > 0 else 0
            
            perimeter = cv2.arcLength(c, True)
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
                
            final_category = category_name
            final_color = color

            if "Clearance" in category_name:
                if solidity > FARM_SOLIDITY_THRESH and area > 1000:
                    final_category = "Agricultural Harvest"
                    final_color = (144, 238, 144) 
            
            if check_linear and final_category != "Agricultural Harvest":
                if extent < active_min_extent:
                    continue
                
                roi = merged_mask[y_int:y_int+h, x_int:x_int+w]
                if detect_linear_features(roi):
                    if aspect_ratio >= MIN_ROAD_ASPECT_RATIO and compactness <= MAX_ROAD_COMPACTNESS:
                        final_category = "Road Development"
                        final_color = (0, 165, 255) 
                    else:
                        final_category = "Construction"
                        final_color = (255, 0, 255) 
                    
            # Topological smoothing optimized for GeoJSON web rendering (0.01)
            epsilon = 0.01 * cv2.arcLength(c, True)
            approx_contour = cv2.approxPolyDP(c, epsilon, True)
                    
            events_list.append({
                "type": final_category,
                "center": (cx, cy),
                "contour": approx_contour,
                "area": int(area),
                "color": final_color,
                "is_global": False
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
            cv2.circle(map_img, e["center"], 4, (0, 0, 0), 1)

    grouped_events = {}
    for e in events:
        grouped_events.setdefault(e["type"], {"color": e["color"], "count": 0, "area": 0, "is_global": e["is_global"]})
        grouped_events[e["type"]]["count"] += 1
        grouped_events[e["type"]]["area"] += e["area"]

    num_items = len(grouped_events) if grouped_events else 1
    panel_height = 90 + (num_items * 35) 
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    
    cv2.putText(panel, f"ENTERPRISE TIMELINE: {d1} to {d2}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(panel, (15, 40), (w - 15, 40), (100, 100, 100), 1)
    
    y_offset = 70

    if not grouped_events:
        cv2.putText(panel, "No significant structural changes detected.", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    else:
        for cat, data in grouped_events.items():
            if data["is_global"]:
                text = f"[FILTERED] {cat}: Map-wide seasonal/atmospheric transition."
                cv2.putText(panel, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            else:
                cv2.rectangle(panel, (15, y_offset - 12), (30, y_offset + 3), data["color"], -1)
                text = f"{cat.upper()}: {data['count']} distinct zones | Total Est. Area: {data['area']} px"
                cv2.putText(panel, text, (45, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            y_offset += 35 

    dashboard = np.vstack((map_img, panel))
    return dashboard

def process_timeframe(d1, d2, layers, target_folder, visuals_dir, dataset_bounds):
    t1_imgs, t2_imgs = {}, {}
    for layer in layers:
        l_folder = os.path.join(target_folder, layer)
        img1 = load_layer(d1, layer, l_folder)
        img2 = load_layer(d2, layer, l_folder)
        if img1 is None or img2 is None:
            return None 
        t1_imgs[layer] = img1
        t2_imgs[layer] = img2
        
    for layer in t1_imgs:
        t1_imgs[layer], t2_imgs[layer] = align_shapes(t1_imgs[layer], t2_imgs[layer])
        
    t1_rgb, t2_rgb = t1_imgs["true_color"], t2_imgs["true_color"]
    t1_nir, t2_nir = t1_imgs["false_color_nir"], t2_imgs["false_color_nir"]
    
    img_h, img_w = t1_rgb.shape[:2]
    total_pixels = img_h * img_w
    img_shape = t1_rgb.shape
    events = []

    cloud_t1 = create_cloud_mask(t1_rgb)
    cloud_t2 = create_cloud_mask(t2_rgb)
    shadow_t1 = create_shadow_mask(t1_rgb)
    shadow_t2 = create_shadow_mask(t2_rgb)
    nodata_t1 = get_nodata_mask(t1_rgb)
    nodata_t2 = get_nodata_mask(t2_rgb)
    
    invalid_pixels = cv2.bitwise_or(cloud_t1, cloud_t2)
    invalid_pixels = cv2.bitwise_or(invalid_pixels, shadow_t1)
    invalid_pixels = cv2.bitwise_or(invalid_pixels, shadow_t2)
    invalid_pixels = cv2.bitwise_or(invalid_pixels, nodata_t1)
    invalid_pixels = cv2.bitwise_or(invalid_pixels, nodata_t2)

    structure_change = get_structural_change_mask(t1_rgb, t2_rgb)

    ndwi_t1 = calculate_ndwi(t1_rgb, t1_nir)
    ndwi_t2 = calculate_ndwi(t2_rgb, t2_nir)
    t1_w = np.uint8((ndwi_t1 > NDWI_WATER_THRESH) * 255)
    t2_w = np.uint8((ndwi_t2 > NDWI_WATER_THRESH) * 255)
    t2_w = cv2.bitwise_and(t2_w, cv2.bitwise_not(shadow_t2)) 
    
    flooded_mask = cv2.bitwise_and(t2_w, cv2.bitwise_not(t1_w))
    dried_mask = cv2.bitwise_and(t1_w, cv2.bitwise_not(t2_w))
    flooded_mask = cv2.bitwise_and(flooded_mask, cv2.bitwise_not(invalid_pixels))
    dried_mask = cv2.bitwise_and(dried_mask, cv2.bitwise_not(invalid_pixels))
    
    events = extract_change_clusters(flooded_mask, "Flooding", (255, 100, 100), events, img_shape) 
    events = extract_change_clusters(dried_mask, "Water Receded", (0, 255, 255), events, img_shape) 

    ndvi_t1 = calculate_ndvi(t1_nir)
    ndvi_t2 = calculate_ndvi(t2_nir)
    
    valid_veg_t1 = ndvi_t1 > NDVI_VEG_THRESH
    valid_veg_t2 = ndvi_t2 > NDVI_VEG_THRESH
    
    ndvi_drop = (ndvi_t1 - ndvi_t2) > NDVI_DROP_MIN
    ndvi_gain = (ndvi_t2 - ndvi_t1) > NDVI_DROP_MIN
    
    raw_veg_loss = np.uint8((valid_veg_t1 & ndvi_drop) * 255)
    raw_veg_gain = np.uint8((valid_veg_t2 & ndvi_gain) * 255)
    
    true_clearance_mask = cv2.bitwise_and(raw_veg_loss, structure_change)
    true_growth_mask = cv2.bitwise_and(raw_veg_gain, structure_change)
    true_clearance_mask = cv2.bitwise_and(true_clearance_mask, cv2.bitwise_not(invalid_pixels))
    true_growth_mask = cv2.bitwise_and(true_growth_mask, cv2.bitwise_not(invalid_pixels))

    t1_g_raw = cv2.cvtColor(t1_rgb, cv2.COLOR_BGR2GRAY)
    t2_g_raw = cv2.cvtColor(t2_rgb, cv2.COLOR_BGR2GRAY)
    diff = cv2.subtract(t2_g_raw, t1_g_raw)
    
    diff_blur = cv2.bilateralFilter(diff, 5, 50, 50)
    _, gated_diff = cv2.threshold(diff_blur, 25, 255, cv2.THRESH_TOZERO)
    
    if np.max(gated_diff) > 0:
        _, albedo_inc = cv2.threshold(gated_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        albedo_inc = np.zeros_like(gated_diff)
    
    built_mask = cv2.bitwise_and(albedo_inc, structure_change)
    built_mask = cv2.bitwise_and(built_mask, cv2.bitwise_not(flooded_mask)) 
    built_mask = cv2.bitwise_and(built_mask, cv2.bitwise_not(raw_veg_loss))
    built_mask = cv2.bitwise_and(built_mask, cv2.bitwise_not(invalid_pixels))

    clearance_ratio = np.sum(true_clearance_mask == 255) / total_pixels
    growth_ratio = np.sum(true_growth_mask == 255) / total_pixels
    built_ratio = np.sum(built_mask == 255) / total_pixels

    skip_clearance = clearance_ratio > GLOBAL_SEASONAL_LIMIT
    skip_growth = growth_ratio > GLOBAL_SEASONAL_LIMIT
    
    # Enforce Strict Mode if any major seasonal transition is occurring
    strict_construction = (built_ratio > GLOBAL_SEASONAL_LIMIT) or skip_clearance or skip_growth

    events = extract_change_clusters(true_clearance_mask, "Land Clearance", (0, 0, 255), events, img_shape, skip=skip_clearance) 
    events = extract_change_clusters(true_growth_mask, "Afforestation / Crop Growth", (0, 255, 0), events, img_shape, skip=skip_growth) 
    events = extract_change_clusters(built_mask, "Construction", (255, 0, 255), events, img_shape, check_linear=True, skip=built_ratio > GLOBAL_SEASONAL_LIMIT, strict_mode=strict_construction)

    if strict_construction and not (skip_clearance or skip_growth):
        events.append({"type": "Global Shift (Albedo Spike)", "center": (0, 0), "contour": None, "area": 0, "color": (128, 128, 128), "is_global": True})

    payload = None
    if events:
        visual_proof = generate_dashboard_image(t2_rgb, events, d1, d2)
        out_path = os.path.join(visuals_dir, f"Polygon_Dashboard_{d1}_to_{d2}.jpg")
        cv2.imwrite(out_path, visual_proof)
        
        geojson_features = []
        for e in events:
            if e["is_global"] or e["contour"] is None:
                continue
                
            coord_list = []
            for point in e["contour"]:
                px, py = int(point[0][0]), int(point[0][1])
                if dataset_bounds:
                    coord_list.append(pixel_to_latlon(px, py, img_w, img_h, dataset_bounds))
                else:
                    coord_list.append([float(px), float(py)])
                
            if len(coord_list) >= 3:
                if coord_list[0] != coord_list[-1]:
                    coord_list.append(coord_list[0])
                    
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coord_list]
                    },
                    "properties": {
                        "event_type": str(e["type"]),
                        "area_pixels": int(e["area"]),
                        "timeframe_start": str(d1),
                        "timeframe_end": str(d2)
                    }
                }
                geojson_features.append(feature)
            
        payload = {"timeframe": f"{d1}_to_{d2}", "features": geojson_features}
        
    # Free memory before thread terminates
    del t1_imgs, t2_imgs, t1_rgb, t2_rgb, t1_nir, t2_nir, cloud_t1, cloud_t2, shadow_t1, shadow_t2
    del invalid_pixels, structure_change, built_mask, true_clearance_mask, true_growth_mask
    gc.collect()
    
    return payload

def run_automated_timeline(target_folder, available_dates):
    layers = ["true_color", "false_color_nir", "agriculture_swir"]
    visuals_dir = os.path.join(target_folder, "visual_reports")
    os.makedirs(visuals_dir, exist_ok=True)
    
    dataset_name = os.path.basename(target_folder)
    dataset_bounds = DATASET_BOUNDS.get(dataset_name, None)
    
    print("\n" + "=" * 70)
    print(f" ENTERPRISE ASYNC TIMELINE PIPELINE INITIALIZED")
    print(f" Georeferencing Status: {'ENABLED (WGS 84)' if dataset_bounds else 'DISABLED (Pixel Space)'}")
    print("=" * 70)
    
    start_time = time.time()
    pairs = [(available_dates[i], available_dates[i+1]) for i in range(len(available_dates)-1)]
    total_pairs = len(pairs)
    
    master_geojson = {
        "type": "FeatureCollection",
        "name": f"Change_Detection_{dataset_name}",
        "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
        "features": []
    }

    safe_workers = max(1, os.cpu_count() - 1)
    completed = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=safe_workers) as executor:
        future_to_pair = {executor.submit(process_timeframe, p[0], p[1], layers, target_folder, visuals_dir, dataset_bounds): p for p in pairs}
        
        for future in concurrent.futures.as_completed(future_to_pair):
            pair = future_to_pair[future]
            completed += 1
            print(f"\r[ {completed} / {total_pairs} ] Processed: {pair[0]} ➔ {pair[1]}", end="", flush=True)
            
            try:
                result = future.result()
                if result:
                    master_geojson["features"].extend(result["features"])
            except Exception as exc:
                print(f"\n[!] Error processing {pair[0]} ➔ {pair[1]}: {exc}")

    master_geojson["features"].sort(key=lambda x: x["properties"]["timeframe_start"])

    json_path = os.path.join(visuals_dir, "timeline_analysis_data.geojson")
    with open(json_path, 'w') as f:
        json.dump(master_geojson, f, indent=4)

    elapsed_time = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"Pipeline Complete in {elapsed_time}s using {safe_workers} CPU cores.")
    print(f"Visual Dashboards & True Polygon GeoJSON Vectors saved in: \n{visuals_dir}")
    print("=" * 70)

def main_menu():
    base_dir = "./satellite_datasets"
    if not os.path.exists(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return

    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    if not folders:
        print(f"No datasets found in '{base_dir}'.")
        return

    print("=" * 60)
    print(" FINALIZED GIS VISUAL TIMELINE GENERATOR")
    print("=" * 60)
    for i, folder in enumerate(folders, 1):
        print(f"  [{i}] {folder}")
    
    try:
        folder_choice = int(input("\nSelect a folder to map: ").strip()) - 1
        if folder_choice < 0 or folder_choice >= len(folders): return
    except ValueError: return
        
    target_directory = os.path.join(base_dir, folders[folder_choice])
    tc_dir = os.path.join(target_directory, "true_color")
    
    if not os.path.exists(tc_dir):
        print("Error: Missing true_color layer. Incompatible dataset.")
        return
        
    files = os.listdir(tc_dir)
    available_dates = sorted(list({re.search(r"(\d{4}-\d{2}-\d{2})", f).group(1) for f in files if re.search(r"(\d{4}-\d{2}-\d{2})", f)}))
    
    if len(available_dates) < 2:
        print("Not enough dates to build a timeline.")
        return

    print(f"\nLocked {len(available_dates)} observation dates. Initializing async pipeline...")
    run_automated_timeline(target_directory, available_dates)

if __name__ == "__main__":
    main_menu()