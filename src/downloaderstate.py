import os
import math
import pystac_client
import planetary_computer
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

print("=" * 60)
print(" State-Level Grid Downloader (Strict On-Image Cloud Filter)")
print("=" * 60)

place_name = input("Enter state or region name (e.g., Goa, Delhi, Chandigarh): ").strip()
output_dir = f"./data_{place_name.lower().replace(' ', '_')}"
os.makedirs(output_dir, exist_ok=True)

print(f"Geocoding '{place_name}' via Nominatim...")
geo_url = "https://nominatim.openstreetmap.org/search"
headers = {'User-Agent': 'SIH-State-Grid-Downloader/1.0'}
params = {'q': place_name, 'format': 'json', 'limit': 1}
res = requests.get(geo_url, params=params, headers=headers).json()

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

target_date_str = input("Enter target timeframe date (YYYY-MM-DD, e.g., 2025-01-01): ").strip()
target_date = datetime.strptime(target_date_str, "%Y-%m-%d")

print("\nSelect Imagery Layer / Band Combination:")
print("  [1] True Color (RGB - Natural Visual)")
print("  [2] False Color / NIR (Vegetation & Water analysis - Bands: B08, B04, B03)")
print("  [3] Agriculture / SWIR (Soil & Crop monitoring - Bands: B11, B8A, B02)")
layer_choice = input("Select option (1, 2, or 3): ").strip()

if layer_choice == "2":
    assets = ["B08", "B04", "B03"]
    layer_name = "False_Color_NIR"
elif layer_choice == "3":
    assets = ["B11", "B8A", "B02"]
    layer_name = "Agriculture_SWIR"
else:
    assets = ["B04", "B03", "B02"]
    layer_name = "True_Color"

step_deg = 0.045
lat_steps = []
curr_lat = lat_min
while curr_lat < lat_max:
    lat_steps.append((curr_lat, min(curr_lat + step_deg, lat_max)))
    curr_lat += step_deg

lon_steps = []
curr_lon = lon_min
while curr_lon < lon_max:
    lon_steps.append((curr_lon, min(curr_lon + step_deg, lon_max)))
    curr_lon += step_deg

total_tiles = len(lat_steps) * len(lon_steps)
print(f"\nGenerated grid layout: {len(lon_steps)} x {len(lat_steps)} = {total_tiles} individual patches.")

# ±1 Year search range
search_start = (target_date - timedelta(days=365)).strftime("%Y-%m-%d")
search_end = (target_date + timedelta(days=365)).strftime("%Y-%m-%d")
time_range = f"{search_start}/{search_end}"

tile_count = 0
for r_idx, (lats, late) in enumerate(lat_steps):
    for c_idx, (lons, lone) in enumerate(lon_steps):
        tile_bbox = [lons, lats, lone, late]
        tile_count += 1
        tile_name = f"tile_r{r_idx}_c{c_idx}"
        
        print(f"\n[{tile_count}/{total_tiles}] Processing patch {tile_name}...")
        
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=tile_bbox,
            datetime=time_range,
            query={"eo:cloud_cover": {"lt": 10}}
        )
        items = list(search.items())
        if not items:
            print(f"  ⚠️ No metadata-clean scenes found for {tile_name}.")
            continue
            
        items.sort(key=lambda item: abs(item.datetime.replace(tzinfo=None) - target_date))
        
        patch_downloaded = False
        for item in items:
            date_str = item.datetime.strftime("%Y-%m-%d")
            filename = f"{date_str}_{tile_name}_{layer_name}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            if os.path.exists(filepath):
                print(f"  -> {filename} already exists locally. Skipping.")
                patch_downloaded = True
                break

            bbox_str = f"{tile_bbox[0]},{tile_bbox[1]},{tile_bbox[2]},{tile_bbox[3]}"
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
                        
                    # 1. Filter out black edge padding (> 20% void)
                    thumb = img.resize((50, 50))
                    pixels = list(thumb.getdata())
                    dark_pixels = sum(1 for p in pixels if sum(p[:3]) < 35)
                    if (dark_pixels / len(pixels)) > 0.20:
                        continue
                        
                    # 2. Strict On-Image Cloud & Haze Pixel Check
                    # Clouds appear as high-luminance, low-saturation white patches (R, G, B all > 200)
                    cloud_pixels = sum(1 for p in pixels if p[0] > 200 and p[1] > 200 and p[2] > 200)
                    cloud_percentage = (cloud_pixels / len(pixels)) * 100
                    
                    if cloud_percentage > 3.0:
                        print(f"  [Rejected] Date {date_str}: Contains visible cloud cover ({round(cloud_percentage, 1)}%). Searching next...")
                        continue
                        
                    img.save(filepath, "JPEG")
                    delta_days = abs((item.datetime.replace(tzinfo=None) - target_date).days)
                    print(f"  ✅ Saved cloud-free patch -> {filename} (Date: {date_str}, Delta: {delta_days} days)")
                    patch_downloaded = True
                    break
            except Exception:
                continue
                
        if not patch_downloaded:
            print(f"  ❌ Could not find a visually cloud-free observation for patch {tile_name} within ±1 year.")

print(f"\nDownload complete! Clean tiles saved in '{output_dir}'.")
