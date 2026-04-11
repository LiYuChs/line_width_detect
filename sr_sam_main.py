from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from segment_anything import SamPredictor, sam_model_registry
from utils import check_path_exists, imwrite_check

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_CHECKPOINT_PATH = "./pretrained_weights/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
RE_ESRGAN_CHECKPOINT_PATH = "./pretrained_weights/RealESRGAN_x4plus.pth"
SR_UPSCALE_FACTOR = 4
LINE_Y_COORDINATES = [407, 418, 429, 441, 452, 463, 474, 485, 496, 507]
CENTER_X = 350

class SRSAMProcessor:
    """純計算引擎：超解析 + SAM 分割模型，負責輸出亞像素寬度計算資料。"""

    def __init__(self, sam_checkpoint_path=SAM_CHECKPOINT_PATH, sam_model_type=SAM_MODEL_TYPE, realesrgan_checkpoint_path=RE_ESRGAN_CHECKPOINT_PATH, upscale_factor=SR_UPSCALE_FACTOR, device=DEVICE, line_y_coordinates=None, center_x=CENTER_X):
        self.sam_checkpoint_path = sam_checkpoint_path
        self.sam_model_type = sam_model_type
        self.realesrgan_checkpoint_path = realesrgan_checkpoint_path
        self.upscale_factor = upscale_factor
        self.device = device
        self.line_y_coordinates = line_y_coordinates or LINE_Y_COORDINATES
        self.center_x = center_x

        if not Path(self.sam_checkpoint_path).exists() or not Path(self.realesrgan_checkpoint_path).exists():
             raise FileNotFoundError("Checkpoints missing for SAM or RealESRGAN.")

        print(f"Executing SR + SAM pipeline on device: {self.device}")
        self._init_sr_model()
        self._init_sam_model()

    def _init_sr_model(self):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(scale=self.upscale_factor, model_path=self.realesrgan_checkpoint_path, model=model, tile=0, device=self.device)

    def _init_sam_model(self):
        sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    @staticmethod
    def _apply_morphology_closing(img_bgr, kernel_size=(3, 3)):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        closed_img = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cv2.cvtColor(closed_img, cv2.COLOR_GRAY2BGR)

    def _build_positive_points(self):
        points = np.array([(self.center_x, y) for y in self.line_y_coordinates], dtype=np.int32)
        return points

    def process(self, image_path):
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        upscaled_img, _ = self.upsampler.enhance(image, outscale=self.upscale_factor)
        enhanced_img_bgr = self._apply_morphology_closing(upscaled_img, kernel_size=(3, 3))
        self.predictor.set_image(cv2.cvtColor(enhanced_img_bgr, cv2.COLOR_BGR2RGB))

        positive_points = self._build_positive_points()
        scaled_points = positive_points * self.upscale_factor
        h_upscale, w_upscale, _ = upscaled_img.shape
        mid_x_upscale = w_upscale // 2

        results = []
        line_segments = []
        all_masks_viz = np.zeros_like(upscaled_img)

        offset_y_hints = [20]
        box_width_half = 400
        box_height_half = 15

        for i in range(len(scaled_points)):
            current_pos_point = scaled_points[i : i + 1]
            x_center = int(current_pos_point[0][0])
            y_center = int(current_pos_point[0][1])

            hints_list = [current_pos_point[0]]
            labels_list = [1]

            for off in offset_y_hints:
                hints_list.append((x_center, y_center - off))
                labels_list.append(0)
                hints_list.append((x_center, y_center + off))
                labels_list.append(0)

            final_points = np.array(hints_list, dtype=np.float32)
            final_labels = np.array(labels_list, dtype=np.int32)
            box_prompt = np.array([x_center - box_width_half, y_center - box_height_half, x_center + box_width_half, y_center + box_height_half])

            masks, _, _ = self.predictor.predict(point_coords=final_points, point_labels=final_labels, box=box_prompt, multimask_output=False)

            best_mask = masks[0]
            mask_u8 = (best_mask.astype(np.uint8) * 255)
            line_segments.append({"id": i + 1, "mask": mask_u8})

            mask_col = best_mask[:, mid_x_upscale]
            line_pixels = np.where(mask_col == 1)[0]

            if len(line_pixels) > 1:
                line_start_upscale = int(line_pixels.min())
                line_end_upscale = int(line_pixels.max())
                integer_width_upscale = line_end_upscale - line_start_upscale + 1
                final_subpixel_width = float(integer_width_upscale / self.upscale_factor)
                results.append({"id": i + 1, "width_px_subpixel": final_subpixel_width})

                color = tuple(np.random.randint(100, 255, 3).tolist())
                colored_mask = np.zeros_like(upscaled_img)
                colored_mask[best_mask == 1] = color
                all_masks_viz = cv2.addWeighted(all_masks_viz, 1, colored_mask, 0.7, 0)

        df = pd.DataFrame(results)

        return {
            "image_path": str(image_path),
            "base_name": image_path.stem,
            "df": df,
            "raw_data": {
                "upscaled_img": upscaled_img,
                "all_masks_viz": all_masks_viz,
                "line_segments": line_segments,
                "results": results,
                "upscale_factor": self.upscale_factor,
                "center_x": self.center_x,
                "line_y_coordinates": self.line_y_coordinates
            }
        }

# ==========================================
# 視覺化與畫圖輔助函式
# ==========================================

def draw_sr_sam_overlay(raw_data, output_path, scale_mm_per_px=0.00363):
    upscaled_img = raw_data["upscaled_img"]
    masks_viz = raw_data["all_masks_viz"]
    results = raw_data["results"]
    upscale_factor = raw_data["upscale_factor"]
    center_x = raw_data["center_x"]
    line_y_coordinates = raw_data["line_y_coordinates"]

    h_upscale, w_upscale, _ = upscaled_img.shape
    mid_x_upscale = w_upscale // 2

    overlay_upscale = cv2.addWeighted(upscaled_img, 0.6, masks_viz, 0.4, 0)
    cv2.line(overlay_upscale, (mid_x_upscale, 0), (mid_x_upscale, h_upscale - 1), (0, 0, 255), 2)

    scaled_points = np.array([(center_x, y) for y in line_y_coordinates], dtype=np.int32) * upscale_factor
    for res in results:
        line_id = res["id"]
        y_scaled = int(scaled_points[line_id - 1][1])
        w_sub = res["width_px_subpixel"]
        w_mm = w_sub * scale_mm_per_px

        cv2.line(overlay_upscale, (mid_x_upscale - 20, y_scaled), (mid_x_upscale + 20, y_scaled), (0, 255, 0), 2)
        cv2.putText(overlay_upscale, f"L{line_id} (SR+Morph+SAM): {w_mm:.4f} mm", (mid_x_upscale + 30, y_scaled + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    check_path_exists(Path(output_path).parent)
    imwrite_check(str(output_path), overlay_upscale)

def export_sr_line_segmentations(raw_data, output_dir):
    upscaled_img = raw_data["upscaled_img"]
    line_segments = raw_data["line_segments"]
    check_path_exists(output_dir)
    
    for seg in line_segments:
        line_id = seg["id"]
        mask = seg["mask"]

        mask_path = output_dir / f"line_{line_id:02d}_mask.png"
        imwrite_check(str(mask_path), mask)

        color_mask = np.zeros_like(upscaled_img)
        color_mask[mask > 0] = (0, 255, 0)
        overlay = cv2.addWeighted(upscaled_img, 0.75, color_mask, 0.25, 0)
        overlay_path = output_dir / f"line_{line_id:02d}_overlay.png"
        imwrite_check(str(overlay_path), overlay)

def export_sr_report(results_list, output_path, scale_mm_per_px=0.00363, known_ground_truths_mm=None):
    known_ground_truths_mm = known_ground_truths_mm or [0.015, 0.016, 0.015, 0.016]
    prev_fwhm_px = [4.15, 4.55, 4.08, 4.27]
    known_widths = list(known_ground_truths_mm)

    while len(known_widths) < len(results_list):
        known_widths.append(None)
    while len(prev_fwhm_px) < len(results_list):
        prev_fwhm_px.append(None)

    rows = []
    for i, res in enumerate(results_list):
        rows.append({
            "Line ID": f"Line {i + 1}",
            "Physical Ground Truth (mm)": known_widths[i],
            "Pixel Width (Previous FWHM)": prev_fwhm_px[i],
            "Pixel Width (Current SR+SAM)": res["width_px_subpixel"],
            "Applied Scale (mm/px)": scale_mm_per_px,
        })

    df = pd.DataFrame(rows)
    df["Verified Phys Width (Current, mm)"] = df["Pixel Width (Current SR+SAM)"] * scale_mm_per_px
    check_path_exists(Path(output_path).parent)
    df.to_excel(str(output_path), index=False)
    return df

# ==========================================
# 批次處理函式
# ==========================================

def run_batch(image_folder, output_filename="SR_SAM_Master_Report.xlsx"):
    processor = SRSAMProcessor()
    image_folder = Path(image_folder)
    image_paths = sorted([p for p in image_folder.iterdir() if p.suffix.lower() in [".jpg", ".png", ".bmp"]])

    all_results = []
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        try:
            result = processor.process(image_path)
            
            # 使用獨立的報表生成邏輯抓出此圖片的 DF (可套用正確的 scale)
            df = export_sr_report(result["raw_data"]["results"], Path.cwd() / "temp.xlsx")
            df.insert(0, "Image Name", result["base_name"])
            all_results.append(df)
            Path("temp.xlsx").unlink(missing_ok=True)
            
        except Exception as err:
            print(f"Error processing {image_path}: {err}")

    if all_results:
        master_df = pd.concat(all_results, ignore_index=True)
        output_path = Path.cwd() / "data" / "sr_sam_result" / output_filename
        check_path_exists(output_path.parent)
        master_df.to_excel(str(output_path), index=False)
        print(f"Master Data successfully exported to {output_path}")

if __name__ == "__main__":
    processor = SRSAMProcessor()
    image_path = Path("./data/measure_01.jpg")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path.cwd() / "data" / "sr_sam_result" / timestamp

    print("Running SR+SAM processor...")
    result = processor.process(image_path)
    
    print("Exporting upscaled image...")
    imwrite_check(str(output_dir / "upscaled_image.jpg"), result["raw_data"]["upscaled_img"])

    print("Generating visual overlays...")
    draw_sr_sam_overlay(result["raw_data"], output_dir / "final_result_subpixel.png")
    export_sr_line_segmentations(result["raw_data"], output_dir / "line_segmentations")

    print("Saving report...")
    export_sr_report(result["raw_data"]["results"], output_dir / "measurement_sr_report.xlsx")