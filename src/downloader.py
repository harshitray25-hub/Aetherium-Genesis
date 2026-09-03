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

while True:
    print("=" * 70)
    print(" Unified Global Sentinel-2 Satellite Dataset Pipeline")
    print("=" * 70)
    print("Choose pipeline mode:")
    print("  [1] State/Region-wide 5x5 km Grid Downloader (Input: State Name, Country)")
    print("  [2] Multi-Temporal Place/BBox Downloader (Input: Place/City/Landmark, Country & Custom Size) [Default 5km Optimal]")
    print("  [3] Exit Pipeline")
    
    pipeline_mode = input("Select option (1, 2, or 3) [Default 1]: ").strip()
    if not pipeline_mode:
        pipeline_mode = "1"

    if pipeline_mode == "3":
        print("Exiting pipeline. Goodbye!")
        break

    if pipeline_mode == "1":
        print("\n--- Mode 1: State/Region 5x5 km Grid Pipeline ---")
        state_input = input("Enter state/region name (e.g., West Bengal, California, Bavaria): ").strip()
        country_input = input("Enter country name (e.g., India, US, Germany): ").strip()
        
        full_query = f"{state_input}, {country_input}"
        clean_place_name = "".join(c if c.isalnum() else "_" for c in state_input.lower())
        place_folder = f"data_{clean_place_name}_5km"

        print(f"\nGeocoding '{full_query}' via Nominatim...")
        geo_url = "https://nominatim.openstreetmap.org/search"
        headers = {'User-Agent': 'SIH-Global-Pipeline/1.0'}
        params = {'q': full_query, 'format': 'json', 'limit': 1}

        try:
            res = requests.get(geo_url, params=params, headers=headers, timeout=10).json()
            if not res or 'boundingbox' not in res[0]:
                print(f"  ⚠️ Could not fetch bounding box for '{full_query}'. Please check your spelling.")
                continue
                
            bb = res[0]['boundingbox']
            lat_min, lat_max = float(bb[0]), float(bb[1])
            lon_min, lon_max = float(bb[2]), float(bb[3])
        except Exception as e:
            print(f"  ❌ Geocoding error for '{full_query}': {e}")
            continue

        target_date = datetime(2025, 1, 1)
        search_start = (target_date - timedelta(days=365)).strftime("%Y-%m-%d")
        search_end = (target_date + timedelta(days=365)).strftime("%Y-%m-%d")
        time_range = f"{search_start}/{search_end}"

        layers = [
            {"name": "true_color", "assets": ["B04", "B03", "B02"]},
            {"name": "false_color_nir", "assets": ["B08", "B04", "B03"]},
            {"name": "agriculture_swir", "assets": ["B11", "B8A", "B02"]}
        ]

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
        print(f"Generated 5x5 km grid layout: {len(lon_steps)} x {len(lat_steps)} = {total_grid_tiles} individual patches.")

        for layer in layers:
            layer_name = layer["name"]
            assets = layer["assets"]
            output_dir = os.path.join(MASTER_DATA_ROOT, place_folder, layer_name)
            os.makedirs(output_dir, exist_ok=True)

            existing_images = [f for f in os.listdir(output_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            existing_count = len(existing_images)

            print(f"\n  📁 Validating & Processing Layer '{layer_name}' (Found {existing_count} existing tiles)...")

            tile_count = 0
            for r_idx, (lats, late) in enumerate(lat_steps):
                for c_idx, (lons, lone) in enumerate(lon_steps):
                    tile_bbox = [lons, lats, lone, late]
                    tile_count += 1
                    tile_name = f"tile_r{r_idx}_c{c_idx}"

                    matching_files = [f for f in os.listdir(output_dir) if tile_name in f and f.endswith(('.jpg', '.jpeg', '.png'))]
                    
                    is_valid = False
                    if matching_files:
                        existing_filepath = os.path.join(output_dir, matching_files[0])
                        try:
                            test_img = Image.open(existing_filepath)
                            thumb = test_img.resize((50, 50))
                            arr = np.array(thumb)
                            test_img.close()
                            
                            dark_pixels = np.sum(np.sum(arr[:, :, :3], axis=2) < 35)
                            row_means = np.mean(arr[:, :, :3], axis=(1, 2))
                            col_means = np.mean(arr[:, :, :3], axis=(0, 2))
                            dead_rows = np.sum(row_means < 35)
                            dead_cols = np.sum(col_means < 35)
                            cloud_pixels = np.sum((arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200))
                            
                            if (dark_pixels / arr.size) <= 0.12 and dead_rows < 8 and dead_cols < 8 and (cloud_pixels / (50 * 50)) * 100 <= 3.0:
                                is_valid = True
                            else:
                                os.remove(existing_filepath)
                        except Exception:
                            if os.path.exists(existing_filepath):
                                os.remove(existing_filepath)

                    if is_valid:
                        continue

                    print(f"    [Fetching missing] Patch {tile_name} ({layer_name})...", end="\r")

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
                            print(f"    [Saved missing] Patch {tile_name} ({layer_name})                    ")

                    except Exception:
                        continue

        print(f"\n🎉 Grid pipeline complete for {full_query}!")

    elif pipeline_mode == "2":
        print("\n--- Mode 2: Multi-Temporal Place/BBox Pipeline ---")
        mode = input("Choose input type:\n  [1] Search by Place Name, Country & Size\n  [2] Enter Direct BBox Coordinates\n  [3] Exit to Main Menu\nSelect option (1, 2, or 3): ").strip()

        if mode == "3":
            continue

        place_folder_name = ""

        if mode == "1":
            place_input = input("Enter place, landmark, or city name (e.g., Kharagpur, Tokyo): ").strip()
            country_input = input("Enter country name (e.g., India, Japan): ").strip()
            full_query = f"{place_input}, {country_input}"
            
            size_input = input("Enter bounding box side length in kilometers [default optimal 5]: ").strip()
            size_km = float(size_input) if size_input else 5.0
            
            print(f"Geocoding '{full_query}'...")
            geo_url = "https://nominatim.openstreetmap.org/search"
            headers = {'User-Agent': 'SIH-Global-Pipeline/1.0'}
            params = {'q': full_query, 'format': 'json', 'limit': 1}
            res = requests.get(geo_url, params=params, headers=headers).json()
            
            if not res:
                print(f"Could not find coordinates for '{full_query}'.")
                continue
            
            center_lat = float(res[0]['lat'])
            center_lon = float(res[0]['lon'])
            print(f"Found center -> Latitude: {center_lat}, Longitude: {center_lon}")
            
            half_km = size_km / 2.0
            d_lat = half_km / 111.0
            d_lon = half_km / (111.0 * math.cos(math.radians(center_lat)))
            
            bbox = [
                center_lon - d_lon,
                center_lat - d_lat,
                center_lon + d_lon,
                center_lat + d_lat
            ]
            
            clean_place_name = "".join(c if c.isalnum() else "_" for c in place_input.lower())
            place_folder_name = f"data_{clean_place_name}_{int(size_km)}km"

        else:
            print("Enter bounding box coordinates [lon_min, lat_min, lon_max, lat_max]:")
            lon_min = float(input("  Min Longitude: ").strip())
            lat_min = float(input("  Min Latitude: ").strip())
            lon_max = float(input("  Max Longitude: ").strip())
            lat_max = float(input("  Max Latitude: ").strip())
            bbox = [lon_min, lat_min, lon_max, lat_max]
            
            center_lat = (lat_min + lat_max) / 2.0
            center_lon = (lon_min + lon_max) / 2.0
            
            print("Reverse geocoding coordinates...")
            rev_url = "https://nominatim.openstreetmap.org/reverse"
            headers = {'User-Agent': 'SIH-Global-Pipeline/1.0'}
            params = {'lat': center_lat, 'lon': center_lon, 'format': 'json'}
            try:
                res = requests.get(rev_url, params=params, headers=headers).json()
                address = res.get('address', {})
                detected_place = address.get('city') or address.get('town') or address.get('county') or address.get('state') or "location"
                clean_place_name = "".join(c if c.isalnum() else "_" for c in detected_place.lower())
            except Exception:
                clean_place_name = "custom_bbox"
                
            place_folder_name = f"data_{clean_place_name}"

        start_date = input("Enter start date (YYYY-MM-DD, e.g., 2020-01-01): ").strip()
        end_date = input("Enter end date (YYYY-MM-DD, e.g., 2025-12-31): ").strip()
        time_range = f"{start_date}/{end_date}"

        target_amount = int(input("Enter exact number of clean image sets required (x): ").strip())

        layer_configs = {
            "true_color": ["B04", "B03", "B02"],
            "false_color_nir": ["B08", "B04", "B03"],
            "agriculture_swir": ["B11", "B8A", "B02"]
        }

        output_dirs = {}
        for layer_suffix in layer_configs.keys():
            dir_path = os.path.join(MASTER_DATA_ROOT, place_folder_name, layer_suffix)
            os.makedirs(dir_path, exist_ok=True)
            output_dirs[layer_suffix] = dir_path

        print(f"\nSearching Planetary Computer archive...")
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=time_range,
            query={"eo:cloud_cover": {"lt": 10}}
        )

        items = list(search.items())
        print(f"Found {len(items)} total candidate scenes. Filtering frames...")

        monthly_dict = {}
        for item in items:
            month_key = item.datetime.strftime("%Y-%m")
            cloud_cover = item.properties.get("eo:cloud_cover", 100)
            
            if month_key not in monthly_dict:
                monthly_dict[month_key] = (cloud_cover, item)
            else:
                if cloud_cover < monthly_dict[month_key][0]:
                    monthly_dict[month_key] = (cloud_cover, item)

        sorted_months = sorted(monthly_dict.keys())
        all_available_items = [monthly_dict[m][1] for m in sorted_months]

        saved_items_count = 0
        tried_items = set()

        while saved_items_count < target_amount:
            existing_dates = set()
            ref_dir = output_dirs["true_color"]
            if os.path.exists(ref_dir):
                for f in os.listdir(ref_dir):
                    if f.endswith(".jpg"):
                        parts = f.split("_")
                        if len(parts) > 0:
                            existing_dates.add(parts[0])

            saved_items_count = len(existing_dates)
            if saved_items_count >= target_amount:
                print(f"\n🎉 Successfully collected all {target_amount} requested image sets!")
                break

            remaining_items = [it for it in all_available_items if it.datetime.strftime("%Y-%m-%d") not in existing_dates and it.id not in tried_items]

            if not remaining_items:
                print(f"\n❌ Warning: Exhausted all available clean candidate frames in the archive. Collected {saved_items_count}/{target_amount} sets.")
                break

            needed_count = target_amount - saved_items_count
            if len(remaining_items) <= needed_count:
                selected_items = remaining_items
            else:
                selected_items = []
                for i in range(needed_count):
                    if needed_count <= 1:
                        idx = 0
                    else:
                        idx = round(i * (len(remaining_items) - 1) / (needed_count - 1))
                    selected_items.append(remaining_items[idx])

            progress_made = False
            for item in selected_items:
                if saved_items_count >= target_amount:
                    break

                date_str = item.datetime.strftime("%Y-%m-%d")
                tried_items.add(item.id)

                all_layers_exist = True
                for layer_suffix in layer_configs.keys():
                    filepath = os.path.join(output_dirs[layer_suffix], f"{date_str}_Sentinel-2_{layer_suffix}.jpg")
                    if not os.path.exists(filepath):
                        all_layers_exist = False
                        break
                        
                if all_layers_exist:
                    continue

                bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
                success_for_this_date = True
                temp_saved_files = []

                for layer_suffix, assets in layer_configs.items():
                    filepath = os.path.join(output_dirs[layer_suffix], f"{date_str}_Sentinel-2_{layer_suffix}.jpg")
                    
                    if os.path.exists(filepath):
                        temp_saved_files.append(filepath)
                        continue

                    crop_url = f"https://planetarycomputer.microsoft.com/api/data/v1/item/bbox/{bbox_str}/800x800.jpg"
                    params = {
                        "collection": "sentinel-2-l2a",
                        "item": item.id,
                        "assets": assets,
                        "color_formula": "Gamma RGB 3.2 Saturation 0.8 Sigmoidal RGB 25 0.35"
                    }
                    
                    try:
                        response = requests.get(crop_url, params=params)
                        if response.status_code == 200:
                            img = Image.open(BytesIO(response.content))
                            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                                img = img.convert("RGB")
                                
                            thumb = img.resize((50, 50))
                            arr = np.array(thumb)
                            dark_pixels = np.sum(np.sum(arr[:, :, :3], axis=2) < 35)
                            total_pixels = 50 * 50
                            
                            row_means = np.mean(arr[:, :, :3], axis=(1, 2))
                            col_means = np.mean(arr[:, :, :3], axis=(0, 2))
                            dead_rows = np.sum(row_means < 35)
                            dead_cols = np.sum(col_means < 35)
                            
                            if (dark_pixels / total_pixels) > 0.15 or dead_rows >= 8 or dead_cols >= 8:
                                success_for_this_date = False
                                break
                                
                            cloud_pixels = np.sum((arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200))
                            if (cloud_pixels / (50 * 50)) * 100 > 3.0:
                                success_for_this_date = False
                                break
                                
                            img.save(filepath, "JPEG")
                            temp_saved_files.append(filepath)
                        else:
                            success_for_this_date = False
                            break
                    except Exception:
                        success_for_this_date = False
                        break

                if success_for_this_date:
                    progress_made = True
                    saved_items_count += 1
                    print(f"[{saved_items_count}/{target_amount}] Saved clean multi-layer suite -> {date_str}")
                else:
                    for p in temp_saved_files:
                        if os.path.exists(p):
                            os.remove(p)

            if not progress_made:
                all_raw_items = list(search.items())
                extra_remaining = [it for it in all_raw_items if it.datetime.strftime("%Y-%m-%d") not in existing_dates and it.id not in tried_items]
                if not extra_remaining:
                    break
                all_available_items = extra_remaining
