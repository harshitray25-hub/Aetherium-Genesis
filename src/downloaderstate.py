import os
import math
import pystac_client
import planetary_computer
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta

MASTER_DATA_ROOT = "satellite_datasets"
os.makedirs(MASTER_DATA_ROOT, exist_ok=True)

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

target_date = datetime(2025, 1, 1)
search_start = (target_date - timedelta(days=365)).strftime("%Y-%m-%d")
search_end = (target_date + timedelta(days=365)).strftime("%Y-%m-%d")
time_range = f"{search_start}/{search_end}"

layers = [
    {"name": "true_color", "assets": ["B04", "B03", "B02"]},
    {"name": "false_color_nir", "assets": ["B08", "B04", "B03"]},
    {"name": "agriculture_swir", "assets": ["B11", "B8A", "B02"]}
]

print("=" * 70)
print(" Interactive Manual Location Pipeline (Strict Strip Void & Clean Replacement)")
print("=" * 70)

place_name = input("Enter state or region name (e.g., Goa, Delhi, Kerala): ").strip()
clean_place_name = "".join(c if c.isalnum() else "_" for c in place_name.lower())
place_folder = f"data_{clean_place_name}_5km"

print(f"Geocoding '{place_name}' via Nominatim...")
geo_url = "https://nominatim.openstreetmap.org/search"
headers = {'User-Agent': 'SIH-Manual-Pipeline/1.0'}
params = {'q': place_name, 'format': 'json', 'limit': 1}

try:
    res = requests.get(geo_url, params=params, headers=headers, timeout=10).json()
    if not res:
        raise ValueError(f"Could not find coordinates for '{place_name}'.")

    if 'boundingbox' in res[0]:
        bb = res[0]['boundingbox']
        lat_min, lat_max = float(bb[0]), float(bb[1])
        lon_min, lon_max = float(bb[2]), float(bb[3])
    else:
        center_lat = float(res[0]['lat'])
        center_lon = float(res[0]['lon'])
        lat_min, lat_max = center_lat - 0.5, center_lat + 0.5
        lon_min, lon_max = center_lon - 0.5, center_lon + 0.5
except Exception as e:
    print(f"❌ Geocoding error for '{place_name}': {e}")
    exit()

step_deg = 0.045
lat_steps, curr_lat = [], lat_min
while curr_lat < lat_max:
    lat_steps.append((curr_lat, min(curr_lat + step_deg, lat_max)))
    curr_lat += step_deg

lon_steps, curr_lon = [], lon_min
while curr_lon < lon_max:
    lon_steps.append((curr_lon, min(curr_lon + step_deg, lon_max)))
    curr_lon += step_deg

total_grid_tiles = len(lat_steps) * len(lon_steps)
print(f"Generated grid layout: {len(lon_steps)} x {len(lat_steps)} = {total_grid_tiles} individual 5x5 km patches.")

for layer in layers:
    layer_name = layer["name"]
    assets = layer["assets"]
    output_dir = os.path.join(MASTER_DATA_ROOT, place_folder, layer_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  📁 Validating & Processing Layer '{layer_name}'...")

    tile_count = 0
    for r_idx, (lats, late) in enumerate(lat_steps):
        for c_idx, (lons, lone) in enumerate(lon_steps):
            tile_bbox = [lons, lats, lone, late]
            tile_count += 1
            tile_name = f"tile_r{r_idx}_c{c_idx}"

            # Find any existing file for this specific tile index (regardless of date prefix)
            matching_files = [f for f in os.listdir(output_dir) if tile_name in f and f.endswith(('.jpg', '.jpeg', '.png'))]
            
            is_valid = False
            if matching_files:
                existing_filepath = os.path.join(output_dir, matching_files[0])
                try:
                    test_img = Image.open(existing_filepath)
                    thumb = test_img.resize((50, 50))
                    arr = np.array(thumb)
                    test_img.close()
                    
                    # 1. Global dark pixel check
                    dark_pixels = np.sum(np.sum(arr[:, :, :3], axis=2) < 35)
                    
                    # 2. Strip-based Void Check (Catches vertical side swaths like left/right black bars)
                    row_means = np.mean(arr[:, :, :3], axis=(1, 2))
                    col_means = np.mean(arr[:, :, :3], axis=(0, 2))
                    dead_rows = np.sum(row_means < 35)
                    dead_cols = np.sum(col_means < 35)
                    
                    # 3. Cloud check
                    cloud_pixels = np.sum((arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200))
                    
                    # If it passes all criteria, keep it
                    if (dark_pixels / arr.size) <= 0.12 and dead_rows < 8 and dead_cols < 8 and (cloud_pixels / (50 * 50)) * 100 <= 3.0:
                        is_valid = True
                    else:
                        # Failed quality checks, remove the bad file
                        os.remove(existing_filepath)
                except Exception:
                    if os.path.exists(existing_filepath):
                        os.remove(existing_filepath)

            if is_valid:
                continue

            print(f"    [Searching/Replacing] Patch {tile_name} ({layer_name})...", end="\r")

            try:
                search = catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=tile_bbox,
                    datetime=time_range,
                    query={"eo:cloud_cover": {"lt": 10}}
                )
                items = list(search.items())
                if not items:
                    continue
                    
                items.sort(key=lambda item: abs(item.datetime.replace(tzinfo=None) - target_date))
                
                patch_downloaded = False
                for item in items:
                    cloud_cover = item.properties.get("eo:cloud_cover", 100.0)
                    if cloud_cover >= 10.0:
                        continue

                    date_str = item.datetime.strftime("%Y-%m-%d")
                    filename = f"{date_str}_{tile_name}_{layer_name}.jpg"
                    filepath = os.path.join(output_dir, filename)

                    bbox_str = f"{tile_bbox[0]},{tile_bbox[1]},{tile_bbox[2]},{tile_bbox[3]}"
                    crop_url = f"https://planetarycomputer.microsoft.com/api/data/v1/item/bbox/{bbox_str}/800x800.jpg"
                    params_api = {
                        "collection": "sentinel-2-l2a",
                        "item": item.id,
                        "assets": assets,
                        "color_formula": "Gamma RGB 3.2 Saturation 0.8 Sigmoidal RGB 25 0.35"
                    }
                    
                    try:
                        response = requests.get(crop_url, params=params_api, timeout=12)
                        if response.status_code == 200:
                            img = Image.open(BytesIO(response.content))
                            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                                img = img.convert("RGB")
                                
                            thumb = img.resize((50, 50))
                            arr = np.array(thumb)
                            
                            # Advanced Strip & Void Validation
                            dark_pixels = np.sum(np.sum(arr[:, :, :3], axis=2) < 35)
                            row_means = np.mean(arr[:, :, :3], axis=(1, 2))
                            col_means = np.mean(arr[:, :, :3], axis=(0, 2))
                            dead_rows = np.sum(row_means < 35)
                            dead_cols = np.sum(col_means < 35)
                            
                            if (dark_pixels / arr.size) > 0.12 or dead_rows >= 8 or dead_cols >= 8:
                                continue
                                
                            cloud_pixels = np.sum((arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200))
                            if (cloud_pixels / (50 * 50)) * 100 > 3.0:
                                continue
                                
                            # Clean up any old mismatched date files for this tile before saving the new one
                            for old_f in matching_files:
                                old_path = os.path.join(output_dir, old_f)
                                if os.path.exists(old_path):
                                    os.remove(old_path)
                                    
                            img.save(filepath, "JPEG")
                            patch_downloaded = True
                            break
                    except requests.exceptions.RequestException:
                        continue
                        
                if patch_downloaded:
                    print(f"    [Cleaned & Replaced] Patch {tile_name} ({layer_name})                    ")

            except Exception:
                continue

print(f"\n🎉 Pipeline complete for {place_name}! All files validated, cleaned, and saved in '{output_dir}'.")
