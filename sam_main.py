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

class SAMProcessor:
    """純計算引擎：直接使用 SAM 進行中心線提示分割，不包含任何檔案儲存與畫圖。"""

    def __init__(self, checkpoint_path=SAM_CHECKPOINT_PATH, model_type=MODEL_TYPE, device=DEVICE, line_y_coordinates=None):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device
        self.line_y_coordinates = line_y_coordinates or LINE_Y_COORDINATES

        print(f"Executing SAM on device: {self.device}")
        print("Loading SAM model into memory...")
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def _build_prompts(self, image_shape):
        h, w = image_shape[:2]
        center_x = w // 2
        valid_y = [y for y in self.line_y_coordinates if 0 <= y < h]

        if not valid_y:
            valid_y = [h // 2]

        points = np.array([[(center_x, y)] for y in valid_y], dtype=np.float32)
        labels = np.ones((len(points), 1), dtype=np.int32)
        return points, labels

    def process(self, image_path):
        image_path = Path(image_path)
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Error: Could not load image {image_path}")

        points, labels = self._build_prompts(img_bgr.shape)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(img_rgb)

        mid_x = img_bgr.shape[1] // 2
        all_masks_viz = np.zeros_like(img_bgr)
        results = []

        print(f"Segmenting {len(points)} lines using direct point prompts...")

        for i in range(len(points)):
            masks, _, _ = self.predictor.predict(
                point_coords=points[i],
                point_labels=labels[i],
                multimask_output=False,
            )

            best_mask = masks[0]
            mask_col = best_mask[:, mid_x]
            line_pixels = np.where(mask_col == 1)[0]

            if len(line_pixels) > 1:
                line_start = int(line_pixels.min())
                line_end = int(line_pixels.max())
                integer_width = line_end - line_start + 1
                center_y = int((line_start + line_end) / 2)
                results.append({"id": i + 1, "center_y": center_y, "width_px_int": integer_width})

                color = tuple(np.random.randint(50, 255, 3).tolist())
                colored_mask = np.zeros_like(img_bgr)
                colored_mask[best_mask == 1] = color
                all_masks_viz = cv2.addWeighted(all_masks_viz, 1, colored_mask, 0.7, 0)
        
        df = pd.DataFrame(results)

        return {
            "image_path": str(image_path),
            "base_name": image_path.stem,
            "df": df,
            "raw_data": {
                "img_bgr": img_bgr,
                "all_masks_viz": all_masks_viz,
                "results": results,
                "mid_x": mid_x
            }
        }

# ==========================================
# 視覺化與畫圖輔助函式
# ==========================================

def draw_sam_overlay(raw_data, output_path):
    img_bgr = raw_data["img_bgr"]
    masks_viz = raw_data["all_masks_viz"]
    results = raw_data["results"]
    mid_x = raw_data["mid_x"]

    overlay = cv2.addWeighted(img_bgr, 0.6, masks_viz, 0.4, 0)
    cv2.line(overlay, (mid_x, 0), (mid_x, img_bgr.shape[0] - 1), (0, 0, 255), 1)

    for res in results:
        y = res["center_y"]
        w = res["width_px_int"]
        cv2.line(overlay, (mid_x - 10, y), (mid_x + 10, y), (0, 255, 0), 1)
        cv2.putText(overlay, f"L{res['id']} SAM: {w}px", (mid_x + 15, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    check_path_exists(Path(output_path).parent)
    imwrite_check(str(output_path), overlay)

# ==========================================
# 批次處理函式
# ==========================================

def run_batch(image_folder, output_filename="SAM_Master_Report.xlsx"):
    processor = SAMProcessor()
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
    img_clean_path = "./data/measure_01.jpg"
    if not Path(SAM_CHECKPOINT_PATH).exists():
        print(f"Error: SAM checkpoint not found at {SAM_CHECKPOINT_PATH}.")
    else:
        output_dir = Path.cwd() / "data" / "sam_result"
        processor = SAMProcessor()
        
        print("Running SAM processor...")
        result = processor.process(img_clean_path)
        
        print("Generating visual plots...")
        output_path = output_dir / f"{result['base_name']}_marked_sam.png"
        draw_sam_overlay(result["raw_data"], output_path)
        print(f"Saved visualization to {output_path}")