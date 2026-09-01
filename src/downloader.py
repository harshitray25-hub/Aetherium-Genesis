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
    print(" Advanced Interactive Sentinel-2 Downloader")
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
    target_amount = int(input("Enter exact number of clean images required (x): ").strip())

    # Layer / Band configuration selection
    print("\nSelect Imagery Layer / Band Combination:")
    print("  [1] True Color (RGB - Natural Visual)")
    print("  [2] False Color / NIR (Vegetation & Water analysis - Bands: B08, B04, B03)")
    print("  [3] Agriculture / SWIR (Soil & Crop monitoring - Bands: B11, B8A, B02)")
    layer_choice = input("Select option (1, 2, or 3): ").strip()

    if layer_choice == "2":
        assets = ["B08", "B04", "B03"]
        layer_suffix = "false_color_nir"
    elif layer_choice == "3":
        assets = ["B11", "B8A", "B02"]
        layer_suffix = "agriculture_swir"
    else:
        assets = ["B04", "B03", "B02"]
        layer_suffix = "true_color"

    output_dir = os.path.join(MASTER_DATA_ROOT, place_folder_name, layer_suffix)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nConfiguration complete.")
    print(f"Target Directory: {output_dir}")
    print(f"Target Clean Images Needed: {target_amount}")

    print("Searching Planetary Computer archive...")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": 10}}
    )

    items = list(search.items())
    print(f"Found {len(items)} total candidate scenes in archive. Validating frames to meet quota...")

    # Group by month to find the cleanest candidate per month, sorted chronologically descending (newest first or oldest first)
    monthly_dict = {}
    for item in items:
        month_key = item.datetime.strftime("%Y-%m")
        cloud_cover = item.properties.get("eo:cloud_cover", 100)
        
        if month_key not in monthly_dict:
            monthly_dict[month_key] = (cloud_cover, item)
        else:
            if cloud_cover < monthly_dict[month_key][0]:
                monthly_dict[month_key] = (cloud_cover, item)

    # Sort available candidate items by date
    sorted_months = sorted(monthly_dict.keys())
    available_items = [monthly_dict[m][1] for m in sorted_months]

    success_count = 0
    saved_items_count = 0

    # Iterate through candidates until we hit our exact target quota (x) or run out of items
    for item in available_items:
        if saved_items_count >= target_amount:
            break

        date_str = item.datetime.strftime("%Y-%m-%d")
        filename = f"{date_str}_Sentinel-2_{layer_suffix}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            print(f"[{saved_items_count + 1}/{target_amount}] Found existing valid file -> {filename}")
            saved_items_count += 1
            continue

        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
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
                dark_pixels = sum(1 for p in thumb.getdata() if sum(p[:3]) < 35)
                total_pixels = 50 * 50
                
                if (dark_pixels / total_pixels) > 0.20:
                    print(f"  [Skipped] {date_str}: Frame contains too much black/empty edge padding. Searching next available...")
                    continue
                    
                img.save(filepath, "JPEG")
                saved_items_count += 1
                print(f"[{saved_items_count}/{target_amount}] Saved clean crop -> {filename}")
            else:
                print(f"  [Error] Failed to fetch {date_str} (Status: {response.status_code})")
        except Exception as e:
            print(f"  [Error] {date_str}: {e}")

    # Strict Quota Enforcement Check
    if saved_items_count < target_amount:
        print(f"\n❌ ERROR: Could not collect the requested {target_amount} images. Only found {saved_items_count} clean frames meeting quality standards in this timeframe/area.")
        print("💡 Tip: Try expanding your date range, increasing the bounding box size, or requesting a smaller number of images.")
    else:
        print(f"\nSuccessfully gathered your requested quota of {target_amount} clean frames in '{output_dir}'.")

    # Ask user whether to continue or close
    choice = input("\nWould you like to download another location/layer?\n  [1] Continue (Run again)\n  [2] Close (Exit)\nSelect option (1 or 2): ").strip()
    if choice != '1':
        print("Exiting downloader. Happy mapping!")
        break
