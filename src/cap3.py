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

print("Initializing Capability 3: Multi-Band Quality Enhancement & Standardization Engine...")
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
    Capability 3: Multi-Band Quality Handling & Image Standardization Engine.
    Extends standardization across all spectral bands (True Color, False Color NIR, and Agriculture SWIR).
    Finds the clearest reference image per band folder, uses it as a master standard, and applies 
    physics-based dehazing, illumination balancing, and histogram matching across all layers.
    """
    def __init__(self, device=device):
        self.device = device

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

        # 1. Physics-based dehazing
        dehazed = self.dark_channel_dehaze(img)

        # 2. PIL Enhancement for balanced contrast and richness
        pil_img = Image.fromarray(cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB))
        pil_img = ImageEnhance.Color(pil_img).enhance(1.15)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.10)
        enhanced_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 3. Histogram matching against the master template image of the same band
        normalized = self.match_histogram(enhanced_bgr, master_template)

        # 4. Edge-preserving smoothing to clean noise while maintaining structure
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
    print(" CAPABILITY 3: MULTI-BAND QUALITY ENHANCEMENT & STANDARDIZATION")
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
        
    target_directory = os.path.join(base_dir, folders[folder_choice])
    
    # Target all three core spectral band directories
    target_bands = {
        "true_color": "true_color_standardized_cleaned",
        "false_color_nir": "false_color_nir_standardized_cleaned",
        "agriculture_swir": "agriculture_swir_standardized_cleaned"
    }

    engine = MultiBandQualityEnhancementPipeline()

    for band_name, output_folder_name in target_bands.items():
        band_dir = os.path.join(target_directory, band_name)
        if not os.path.exists(band_dir):
            print(f"\n[-] Skipping band '{band_name}' (Directory not found).")
            continue

        image_files = sorted(glob.glob(os.path.join(band_dir, "*.jpg")) + glob.glob(os.path.join(band_dir, "*.png")))
        if not image_files:
            print(f"\n[-] Skipping band '{band_name}' (No images found).")
            continue

        print(f"\n------------------------------------------------------------")
        print(f" Processing Band: [{band_name.upper()}] ({len(image_files)} images)")
        print(f"------------------------------------------------------------")

        print("Evaluating images to find the clearest master reference standard for this band...")
        best_image_path = max(image_files, key=lambda p: engine.evaluate_image_clarity(p))
        master_template = cv2.imread(best_image_path)
        print(f"[*] Master Reference Standard for {band_name}: {os.path.basename(best_image_path)}")

        output_processed_dir = os.path.join(target_directory, output_folder_name)
        os.makedirs(output_processed_dir, exist_ok=True)

        for path in image_files:
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
    print(f"Capability 3 Multi-Band Standardization Complete!")
    print(f"All available bands successfully cleaned, enhanced, and saved.")
    print("=" * 60)

if __name__ == "__main__":
    interactive_multiband_standardization_menu()