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

print("Initializing Capability 3: Multi-Band Quality Gating & Perimeter Border Check Engine...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

print(f"Loading RemoteCLIP ({model_name}) weights offline for clarity assessment...")
try:
    ckpt_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", local_files_only=True)
except Exception:
    ckpt_path = f"checkpoints/RemoteCLIP-{model_name}.pt"

if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
else:
    print(f"Warning: Checkpoint file not found at {ckpt_path}. Ensure weights are downloaded locally.")

model = model.to(device).eval()
print("Model loaded locally and ready!\n")

class MultiBandQualityEnhancementPipeline:
    """
    Capability 3: Multi-Band Quality Handling & Standardization Engine.
    Strictly checks for cloud cover and actual black border edges/nodata padding 
    around the image frame perimeter, structuring outputs neatly into 
    'satellite_datasets_cleaned/data_placename_size/[layers]'.
    """
    def __init__(self, device=device):
        self.device = device

    def check_cloud_rejection(self, img_bgr, max_cloud_pct=15.0):
        """Detects and measures cloud cover percentage using HSV thresholds."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        bright = v > 180
        low_sat = s < 45
        cloud_pixels = np.sum(bright & low_sat)
        total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
        pct = (cloud_pixels / total_pixels) * 100
        return pct > max_cloud_pct, pct

    def check_black_borders_rejection(self, img_bgr, border_thickness_pct=0.05, max_edge_black_pct=25.0):
        """
        Checks specifically if the outer edges (borders) are completely black/nodata 
        horizontally and vertically, avoiding false rejections from dark urban pixels inside the map.
        """
        h, w = img_bgr.shape[:2]
        tb = int(h * border_thickness_pct)
        lr = int(w * border_thickness_pct)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Extract outer border frames (top, bottom, left, right strips)
        top_strip = gray[:tb, :]
        bottom_strip = gray[h-tb:, :]
        left_strip = gray[:, :lr]
        right_strip = gray[:, w-lr:]

        borders = np.concatenate([top_strip.flatten(), bottom_strip.flatten(), left_strip.flatten(), right_strip.flatten()])
        black_border_pixels = np.sum(borders < 15)
        border_pct = (black_border_pixels / borders.size) * 100

        return border_pct > max_edge_black_pct, border_pct

    def validate_frame(self, image_path):
        """Runs strict rejection checks for clouds and actual outer black borders."""
        img = cv2.imread(image_path)
        if img is None:
            return True, "Error: Failed to read image file."

        is_cloudy, cloud_pct = self.check_cloud_rejection(img)
        if is_cloudy:
            return True, f"Rejected: Excessive cloud cover detected ({cloud_pct:.1f}% > 15%)."

        has_black_edges, edge_black_pct = self.check_black_borders_rejection(img)
        if has_black_edges:
            return True, f"Rejected: Excessive black border padding detected ({edge_black_pct:.1f}% > 25% on edges)."

        return False, "Passed quality gate."

    def dark_channel_dehaze(self, img_bgr, omega=0.82, patch_size=15):
        """Applies Dark Channel Prior (DCP) physics-based dehazing to strip out atmospheric smog/fog."""
        img_float = img_bgr.astype(np.float64) / 255.0
        min_channel = np.min(img_float, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        flat_dark = dark_channel.flatten()
        flat_img = img_float.reshape(-1, 3)
        num_pixels = flat_dark.size
        num_search = max(int(num_pixels * 0.001), 1)
        
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
        """Forces source image color distribution to match template image for multi-temporal consistency."""
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
        """Scores image clarity using Laplacian variance (sharpness) and CLIP ground confidence."""
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
        """Enhances, dehazes, and standardizes an individual multi-band image against its band reference."""
        img = cv2.imread(image_path)
        if img is None:
            return None

        dehazed = self.dark_channel_dehaze(img)

        pil_img = Image.fromarray(cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB))
        pil_img = ImageEnhance.Color(pil_img).enhance(1.15)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.10)
        enhanced_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        normalized = self.match_histogram(enhanced_bgr, master_template)
        cleaned_output = cv2.bilateralFilter(normalized, d=5, sigmaColor=30, sigmaSpace=30)
        return cleaned_output

def interactive_multiband_standardization_menu():
    base_dir = "./satellite_datasets"
    if not os.path.exists(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return

    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    if not folders:
        print(f"No datasets found in '{base_dir}'.")
        return

    print("=" * 60)
    print(" CAPABILITY 3: PERIMETER BORDER CHECK & STANDARDIZATION")
    print("=" * 60)
    for i, folder in enumerate(folders, 1):
        print(f"   [{i}] {folder}")
    
    try:
        folder_choice = int(input("\nSelect dataset region folder to standardize: ").strip()) - 1
        if not (0 <= folder_choice < len(folders)):
            print("Invalid selection.")
            return
    except ValueError:
        print("Invalid input.")
        return
        
    region_name = folders[folder_choice]
    target_directory = os.path.join(base_dir, region_name)
    
    master_cleaned_root = "./satellite_datasets_cleaned"
    region_output_root = os.path.join(master_cleaned_root, region_name)
    os.makedirs(region_output_root, exist_ok=True)

    target_bands = ["true_color", "false_color_nir", "agriculture_swir"]
    engine = MultiBandQualityEnhancementPipeline()

    for band_name in target_bands:
        band_dir = os.path.join(target_directory, band_name)
        if not os.path.exists(band_dir):
            print(f"\n[-] Skipping band '{band_name}' (Directory not found).")
            continue

        image_files = sorted(glob.glob(os.path.join(band_dir, "*.jpg")) + glob.glob(os.path.join(band_dir, "*.png")))
        if not image_files:
            print(f"\n[-] Skipping band '{band_name}' (No images found).")
            continue

        print(f"\n------------------------------------------------------------")
        print(f" Processing Band: [{band_name.upper()}] ({len(image_files)} total files)")
        print(f"------------------------------------------------------------")

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

        if not valid_images:
            print(f"  ⚠️ No images passed quality gates for band '{band_name}'.")
            continue

        print("\nEvaluating surviving images to find the master reference standard...")
        best_image_path = max(valid_images, key=lambda p: engine.evaluate_image_clarity(p))
        master_template = cv2.imread(best_image_path)
        print(f"[*] Master Reference Standard for {band_name}: {os.path.basename(best_image_path)}")

        print(f"\nEnhancing and standardizing {len(valid_images)} valid images...")
        for path in valid_images:
            filename = os.path.basename(path)
            print(f"  -> Processing & cleaning: {filename}")
            
            cleaned_img = engine.process_and_standardize_image(path, master_template)
            if cleaned_img is not None:
                out_path = os.path.join(output_processed_dir, filename)
                cv2.imwrite(out_path, cleaned_img)
                print(f"     [+] Enhanced, standardized & saved successfully.")
            else:
                print(f"     [!] Failed to process image.")

    print("\n" + "=" * 60)
    print(f"Capability 3 Restructuring & Perimeter Border Check Complete!")
    print(f"Cleaned multi-band layers saved under: '{region_output_root}'")
    print("=" * 60)

if __name__ == "__main__":
    interactive_multiband_standardization_menu()
