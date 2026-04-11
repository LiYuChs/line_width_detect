from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from segment_anything import SamPredictor, sam_model_registry
from utils import check_path_exists, imwrite_check

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_CHECKPOINT_PATH = "./pretrained_weights/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
LINE_Y_COORDINATES = [407, 418, 429, 441, 452, 463, 474, 485, 496, 507]

class OptimizedSAMProcessor:
    """純計算引擎：ROI + upscale + morphology 精煉的 SAM，不包含畫圖與檔案儲存。"""

    def __init__(self, checkpoint_path=SAM_CHECKPOINT_PATH, model_type=MODEL_TYPE, device=DEVICE, roi_start=350, roi_end=550, line_y_coordinates=None):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device
        self.roi_start = roi_start
        self.roi_end = roi_end
        self.line_y_coordinates = line_y_coordinates or LINE_Y_COORDINATES

        print("Loading Optimized SAM model into memory...")
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def _build_points_for_image(self, shape):
        h, w = shape[:2]
        center_x = w // 2
        valid_y = [y for y in self.line_y_coordinates if 0 <= y < h]

        if not valid_y:
            fallback_y = max(0, min(h - 1, (self.roi_start + self.roi_end) // 2))
            valid_y = [fallback_y]

        points = np.array([(center_x, y) for y in valid_y], dtype=np.int32)
        return points

    def process(self, image_path, upscale_factor=4):
        image_path = Path(image_path)
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        h, w, _ = img_bgr.shape
        roi_start = max(0, min(self.roi_start, h - 1))
        roi_end = max(roi_start + 1, min(self.roi_end, h))
        mid_x = w // 2

        positive_points = self._build_points_for_image(img_bgr.shape)

        roi_img = img_bgr[roi_start:roi_end, :]
        crop_h, crop_w, _ = roi_img.shape
        upscaled_roi = cv2.resize(roi_img, (crop_w * upscale_factor, crop_h * upscale_factor), interpolation=cv2.INTER_LINEAR)

        self.predictor.set_image(cv2.cvtColor(upscaled_roi, cv2.COLOR_BGR2RGB))

        sam_mask_viz = np.zeros_like(img_bgr)
        morph_union = np.zeros((crop_h * upscale_factor, crop_w * upscale_factor), dtype=np.uint8)

        scaled_mid_x = int(mid_x * upscale_factor)
        scaled_mid_x = max(0, min(scaled_mid_x, morph_union.shape[1] - 1))

        results = []
        offset_x_hints = [25]
        rel_points = positive_points - np.array([0, roi_start], dtype=np.int32)
        scaled_points = rel_points * np.array([upscale_factor, upscale_factor], dtype=np.int32)

        for i in range(len(scaled_points)):
            current_pos = scaled_points[i]
            hints = [tuple(current_pos)]
            labels = [1]

            for off in offset_x_hints:
                scaled_off = off * upscale_factor
                hints.append((current_pos[0] - scaled_off, current_pos[1]))
                labels.append(0)
                hints.append((current_pos[0] + scaled_off, current_pos[1]))
                labels.append(0)

            point_coords = np.array(hints, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)

            masks, _, _ = self.predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=False)

            best_mask = masks[0].astype(np.uint8)
            refined_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            morph_union = np.maximum(morph_union, refined_mask)

            mask_col = refined_mask[:, scaled_mid_x]
            line_pixels = np.where(mask_col == 1)[0]

            if len(line_pixels) > 1:
                l_start = int(line_pixels.min())
                l_end = int(line_pixels.max())
                width_scaled = l_end - l_start + 1
                width_px = int(round(width_scaled / upscale_factor))
                center_y_abs = int(round((l_start + l_end) / 2 / upscale_factor)) + roi_start
                results.append({"id": i + 1, "center_y_abs": center_y_abs, "width_px_int": width_px})

            mask_rescaled = cv2.resize(refined_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            mask_full = np.zeros((h, w), dtype=np.uint8)
            mask_full[roi_start:roi_end, :] = mask_rescaled

            color = tuple(np.random.randint(100, 255, 3).tolist())
            color_mask = np.zeros_like(img_bgr)
            color_mask[mask_full == 1] = color
            sam_mask_viz = cv2.addWeighted(sam_mask_viz, 1, color_mask, 0.7, 0)
            
        df = pd.DataFrame(results)

        return {
            "image_path": str(image_path),
            "base_name": image_path.stem,
            "df": df,
            "raw_data": {
                "img_bgr": img_bgr,
                "sam_mask_viz": sam_mask_viz,
                "morph_union": morph_union,
                "upscaled_roi": upscaled_roi,
                "results": results,
                "roi_start": roi_start,
                "roi_end": roi_end,
                "mid_x": mid_x,
                "h": h,
                "w": w
            }
        }

# ==========================================
# 視覺化與畫圖輔助函式
# ==========================================

def draw_optimized_sam_overlays(raw_data, sam_output_path, morph_output_path):
    img_bgr = raw_data["img_bgr"]
    sam_mask_viz = raw_data["sam_mask_viz"]
    morph_union = raw_data["morph_union"]
    upscaled_roi = raw_data["upscaled_roi"]
    results = raw_data["results"]
    roi_start = raw_data["roi_start"]
    roi_end = raw_data["roi_end"]
    mid_x = raw_data["mid_x"]
    w = raw_data["w"]
    h = raw_data["h"]

    # SAM Overlay
    sam_overlay = cv2.addWeighted(img_bgr, 0.6, sam_mask_viz, 0.4, 0)
    cv2.rectangle(sam_overlay, (0, roi_start), (w - 1, roi_end - 1), (0, 255, 255), 2)
    cv2.line(sam_overlay, (mid_x, 0), (mid_x, h - 1), (0, 0, 255), 1)

    for res in results:
        y = res["center_y_abs"]
        width = res["width_px_int"]
        cv2.line(sam_overlay, (mid_x - 10, y), (mid_x + 10, y), (0, 255, 0), 1)
        cv2.putText(sam_overlay, f"L{res['id']} SAM: {width}px", (mid_x + 15, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    check_path_exists(Path(sam_output_path).parent)
    imwrite_check(str(sam_output_path), sam_overlay)

    # Morph Overlay
    morph_binary_vis = cv2.cvtColor((morph_union * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    morph_overlay = cv2.addWeighted(upscaled_roi, 0.65, morph_binary_vis, 0.35, 0)
    
    check_path_exists(Path(morph_output_path).parent)
    imwrite_check(str(morph_output_path), morph_overlay)

# ==========================================
# 批次處理函式
# ==========================================

def run_batch(image_folder, output_filename="OptimizedSAM_Master_Report.xlsx"):
    processor = OptimizedSAMProcessor()
    image_folder = Path(image_folder)
    image_paths = sorted([p for p in image_folder.iterdir() if p.suffix.lower() in [".jpg", ".png", ".bmp"]])

    all_results = []
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        try:
            result = processor.process(image_path)
            df = result["df"]
            df.insert(0, "Image Name", result["base_name"])
            all_results.append(df)
        except Exception as err:
            print(f"Error processing {image_path}: {err}")

    if all_results:
        master_df = pd.concat(all_results, ignore_index=True)
        output_path = Path.cwd() / "data" / "sam_result" / output_filename
        check_path_exists(output_path.parent)
        master_df.to_excel(str(output_path), index=False)
        print(f"Master Data successfully exported to {output_path}")

if __name__ == "__main__":
    if not Path(SAM_CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {SAM_CHECKPOINT_PATH}")

    output_dir = Path.cwd() / "data" / "opt_sam_result"
    processor = OptimizedSAMProcessor()
    
    print("Running Optimized SAM processor...")
    result = processor.process("./data/measure_01.jpg")
    
    print("Generating visual plots...")
    sam_out = output_dir / f"{result['base_name']}_optimized_sam.png"
    morph_out = output_dir / f"{result['base_name']}_optimized_morph.png"
    
    draw_optimized_sam_overlays(result["raw_data"], sam_out, morph_out)
    print(f"Saved: {sam_out}\nSaved: {morph_out}")