import os
import math
import pystac_client
import planetary_computer
import requests
from PIL import Image
from io import BytesIO

# Master parent directory where all location datasets will be organized
MASTER_DATA_ROOT = "satellite_datasets"
os.makedirs(MASTER_DATA_ROOT, exist_ok=True)

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

while True:
    print("=" * 60)
    print(" Advanced Interactive Sentinel-2 Downloader (Multi-Layer)")
    print("=" * 60)

    # Mode selection: Place Name with Size in KM vs Direct BBox Coordinates
    mode = input("Choose input type:\n  [1] Search by Place Name & Size (in km)\n  [2] Enter Direct BBox Coordinates\nSelect option (1 or 2): ").strip()

    place_folder_name = ""

    if mode == "1":
        place_name = input("Enter place name (e.g., Navi Mumbai Airport, Sector 9 Gurgaon): ").strip()
        
        size_input = input("Enter bounding box side length in kilometers [default 3]: ").strip()
        size_km = float(size_input) if size_input else 3.0
        
        print(f"Geocoding '{place_name}'...")
        geo_url = "https://nominatim.openstreetmap.org/search"
        headers = {'User-Agent': 'SIH-Satellite-Downloader/1.0'}
        params = {'q': place_name, 'format': 'json', 'limit': 1}
        res = requests.get(geo_url, params=params, headers=headers).json()
        
        if not res:
            raise ValueError(f"Could not find coordinates for '{place_name}'. Please try direct coordinates mode.")
        
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
        
        clean_place_name = "".join(c if c.isalnum() else "_" for c in place_name.lower())
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
        
        print("Reverse geocoding coordinates to determine place name...")
        rev_url = "https://nominatim.openstreetmap.org/reverse"
        headers = {'User-Agent': 'SIH-Satellite-Downloader/1.0'}
        params = {'lat': center_lat, 'lon': center_lon, 'format': 'json'}
        try:
            res = requests.get(rev_url, params=params, headers=headers).json()
            address = res.get('address', {})
            detected_place = address.get('city') or address.get('town') or address.get('county') or address.get('state') or "location"
            clean_place_name = "".join(c if c.isalnum() else "_" for c in detected_place.lower())
        except Exception:
            clean_place_name = "custom_bbox"
            
        place_folder_name = f"data_{clean_place_name}"

    # Time period configuration
    start_date = input("Enter start date (YYYY-MM-DD, e.g., 2023-01-01): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD, e.g., 2026-12-31): ").strip()
    time_range = f"{start_date}/{end_date}"

    # Target exact amount configuration (x images)
    target_amount = int(input("Enter exact number of clean image sets required (x): ").strip())

    # Master Layer Dictionary mapping the folder name to its specific satellite bands
    layer_configs = {
        "true_color": ["B04", "B03", "B02"],
        "false_color_nir": ["B08", "B04", "B03"],
        "agriculture_swir": ["B11", "B8A", "B02"]
    }

    # Generate all three sub-directories automatically
    output_dirs = {}
    for layer_suffix in layer_configs.keys():
        dir_path = os.path.join(MASTER_DATA_ROOT, place_folder_name, layer_suffix)
        os.makedirs(dir_path, exist_ok=True)
        output_dirs[layer_suffix] = dir_path

    print(f"\nConfiguration complete.")
    print(f"Target Parent Directory: {os.path.join(MASTER_DATA_ROOT, place_folder_name)}")
    print(f"Target Clean Image Sets Needed: {target_amount} (Extracting 3 layers per set)")

    print("Searching Planetary Computer archive...")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": 10}}
    )

    items = list(search.items())
    print(f"Found {len(items)} total candidate scenes in archive. Validating frames to meet quota...")

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
    available_items = [monthly_dict[m][1] for m in sorted_months]

    saved_items_count = 0

    for item in available_items:
        if saved_items_count >= target_amount:
            break

        date_str = item.datetime.strftime("%Y-%m-%d")
        
        # Check if all 3 layers for this date already exist locally
        all_layers_exist = True
        for layer_suffix in layer_configs.keys():
            filepath = os.path.join(output_dirs[layer_suffix], f"{date_str}_Sentinel-2_{layer_suffix}.jpg")
            if not os.path.exists(filepath):
                all_layers_exist = False
                break
                
        if all_layers_exist:
            print(f"[{saved_items_count + 1}/{target_amount}] Found existing valid files for all layers -> {date_str}")
            saved_items_count += 1
            continue

        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        
        # Keep track of successful downloads in this iteration in case one fails
        success_for_this_date = True
        temp_saved_files = []

        for layer_suffix, assets in layer_configs.items():
            filepath = os.path.join(output_dirs[layer_suffix], f"{date_str}_Sentinel-2_{layer_suffix}.jpg")
            
            # Skip if just this specific layer already downloaded in a previous broken run
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
                        
                    # Artifact gating applied to each layer
                    thumb = img.resize((50, 50))
                    dark_pixels = sum(1 for p in thumb.getdata() if sum(p[:3]) < 35)
                    total_pixels = 50 * 50
                    
                    if (dark_pixels / total_pixels) > 0.20:
                        print(f"  [Skipped] {date_str}: {layer_suffix} frame contains too much black/empty edge padding.")
                        success_for_this_date = False
                        break
                        
                    img.save(filepath, "JPEG")
                    temp_saved_files.append(filepath)
                else:
                    print(f"  [Error] Failed to fetch {date_str} {layer_suffix} (Status: {response.status_code})")
                    success_for_this_date = False
                    break
            except Exception as e:
                print(f"  [Error] {date_str} {layer_suffix}: {e}")
                success_for_this_date = False
                break

        if success_for_this_date:
            saved_items_count += 1
            print(f"[{saved_items_count}/{target_amount}] Saved clean multi-layer suite -> {date_str}")
        else:
            # Clean up incomplete sets if one layer fails artifact gating
            for p in temp_saved_files:
                if os.path.exists(p):
                    os.remove(p)
            print(f"  [Skipped] {date_str}: Failed to gather all 3 layers cleanly. Moving to next date...")

    if saved_items_count < target_amount:
        print(f"\n❌ ERROR: Could not collect the requested {target_amount} image sets. Only found {saved_items_count} clean frames meeting quality standards.")
        print("💡 Tip: Try expanding your date range or increasing the bounding box size.")
    else:
        print(f"\nSuccessfully gathered your requested quota of {target_amount} multi-layer frame sets.")

    choice = input("\nWould you like to download another location?\n  [1] Continue (Run again)\n  [2] Close (Exit)\nSelect option (1 or 2): ").strip()
    if choice != '1':
        print("Exiting downloader. Happy mapping!")
        break
