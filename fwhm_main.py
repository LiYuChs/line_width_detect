import os
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import find_peaks, peak_widths

from utils import check_path_exists, imwrite_check


class FWHMProcessor:
    """純計算引擎：只負責 FWHM 與 Hough Transform 運算，不包含任何畫圖與檔案 IO。"""

    def __init__(
        self,
        roi_start=350,
        roi_end=550,
        peak_height=40,
        peak_prominence=15,
        peak_distance=5,
        known_ground_truths_mm=None,
    ):
        self.roi_start = roi_start
        self.roi_end = roi_end
        self.peak_height = peak_height
        self.peak_prominence = peak_prominence
        self.peak_distance = peak_distance
        self.known_ground_truths_mm = known_ground_truths_mm or [0.015, 0.016, 0.015, 0.016]

    def _extract_fwhm_widths(self, img_gray):
        h, w = img_gray.shape
        mid_x = w // 2

        col_intensities = img_gray[:, mid_x]
        section = col_intensities[self.roi_start:self.roi_end]

        peaks, _ = find_peaks(
            section,
            height=self.peak_height,
            prominence=self.peak_prominence,
            distance=self.peak_distance,
        )
        pixel_widths, _, left_ips, right_ips = peak_widths(section, peaks, rel_height=0.5)

        abs_peaks = peaks + self.roi_start
        abs_left_ips = left_ips + self.roi_start
        abs_right_ips = right_ips + self.roi_start

        return {
            "h": h,
            "w": w,
            "mid_x": mid_x,
            "peaks": abs_peaks,
            "pixel_widths": pixel_widths,
            "left_ips": abs_left_ips,
            "right_ips": abs_right_ips,
            "section": section,
        }

    def _calculate_tilt_angle(self, img_gray, mid_x):
        roi_right = img_gray[:, mid_x:]
        edges = cv2.Canny(roi_right, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/1800, 
            threshold=100, 
            minLineLength=80, 
            maxLineGap=25
        )

        angles_rad = []
        angles_deg = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                angle_rad = np.arctan((y2 - y1) / (x2 - x1))
                angles_rad.append(angle_rad)
                angles_deg.append(np.degrees(angle_rad))

        median_angle_rad = np.median(angles_rad) if angles_rad else 0.0

        return {
            "median_angle_rad": median_angle_rad,
            "edges": edges,
            "lines": lines,
            "angles_deg": angles_deg
        }

    def calculate_rotation_angle(self, img_gray, mid_x):
        """回傳旋轉角度（rad/deg）以及 Hough 偵測除錯資料。"""
        hough_data = self._calculate_tilt_angle(img_gray, mid_x)
        angle_rad = float(hough_data["median_angle_rad"])
        angle_deg = float(np.degrees(angle_rad))
        return angle_rad, angle_deg, hough_data

    @staticmethod
    def _calculate_scale(pixel_widths, known_physical_widths):
        if len(pixel_widths) == 0:
            return 0.0

        usable = min(len(pixel_widths), len(known_physical_widths))
        if usable == 0:
            return 0.0

        scales = [known_physical_widths[i] / pixel_widths[i] for i in range(usable)]
        return float(sum(scales) / len(scales))

    def process(self, image_path):
        """核心處理流程，只回傳計算結果與供畫圖使用的 raw data。"""
        image_path = Path(image_path)
        img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Error: Could not load image {image_path}")

        extracted = self._extract_fwhm_widths(img_gray)
        tilt_angle_rad, tilt_angle_deg, hough_data = self.calculate_rotation_angle(img_gray, extracted["mid_x"])
        compensation_factor = np.cos(tilt_angle_rad)

        corrected_pixel_widths = extracted["pixel_widths"] * compensation_factor
        scale_mm_per_px = self._calculate_scale(corrected_pixel_widths[:4], self.known_ground_truths_mm)
        physical_widths_mm = [round(pw * scale_mm_per_px, 3) for pw in corrected_pixel_widths]

        base_name = image_path.stem
        
        df = pd.DataFrame(
            {
                "Image Name": [base_name] * len(extracted["peaks"]),
                "Line ID": [f"Line {i + 1}" for i in range(len(extracted["peaks"]))],
                "Y Coordinate (Pixel)": extracted["peaks"],
                "Raw Pixel Width (FWHM)": extracted["pixel_widths"],
                "Tilt Angle (Deg)": [tilt_angle_deg] * len(extracted["peaks"]),
                "Corrected Pixel Width": corrected_pixel_widths,
                "Applied Scale (mm/px)": [scale_mm_per_px] * len(extracted["peaks"]),
                "Physical Width (mm)": physical_widths_mm,
            }
        )

        return {
            "image_path": str(image_path),
            "base_name": base_name,
            "df": df,
            "raw_data": {
                "extracted": extracted,
                "hough_data": hough_data,
                "tilt_angle_deg": tilt_angle_deg,
                "physical_widths_mm": physical_widths_mm,
                "roi_start": self.roi_start
            }
        }

# ==========================================
# 視覺化與畫圖輔助函式 (獨立於 Processor 之外)
# ==========================================

def draw_results_image(image_path, raw_data, output_path):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return

    mid_x = raw_data["extracted"]["mid_x"]
    peaks = raw_data["extracted"]["peaks"]
    left_ips = raw_data["extracted"]["left_ips"]
    right_ips = raw_data["extracted"]["right_ips"]
    physical_widths_mm = raw_data["physical_widths_mm"]
    tilt_angle_deg = raw_data["tilt_angle_deg"]
    
    h, _, _ = img_bgr.shape
    cv2.line(img_bgr, (mid_x, 0), (mid_x, h - 1), (0, 0, 255), 1)

    for i in range(len(peaks)):
        y_peak = int(round(peaks[i]))
        y_left = int(round(left_ips[i]))
        y_right = int(round(right_ips[i]))

        cv2.line(img_bgr, (mid_x - 10, y_peak), (mid_x + 10, y_peak), (0, 255, 0), 1)
        cv2.line(img_bgr, (mid_x + 25, y_left), (mid_x + 25, y_right), (0, 255, 0), 2)

        text = f"L{i + 1}: {physical_widths_mm[i]:.3f}mm"
        cv2.putText(img_bgr, text, (mid_x + 35, y_peak + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    info_text = f"Tilt Angle: {tilt_angle_deg:.2f} deg"
    cv2.putText(img_bgr, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    imwrite_check(str(output_path), img_bgr)

def plot_intensity_profile(raw_data, output_path):
    section_intensities = raw_data["extracted"]["section"]
    roi_start = raw_data["roi_start"]
    peaks = raw_data["extracted"]["peaks"]
    left_ips = raw_data["extracted"]["left_ips"]
    right_ips = raw_data["extracted"]["right_ips"]
    tilt_angle_deg = raw_data.get("tilt_angle_deg", 0.0)

    y_coords = np.arange(roi_start, roi_start + len(section_intensities))
    rel_peaks = peaks - roi_start

    plt.figure(figsize=(10, 4))
    plt.plot(y_coords, section_intensities, color="black", label="Pixel Intensity (0-255)")
    plt.plot(peaks, section_intensities[rel_peaks.astype(int)], "x", color="red", markersize=8, label="Detected Center")

    width_heights = section_intensities[rel_peaks.astype(int)] - (
        section_intensities[rel_peaks.astype(int)] - np.min(section_intensities)
    ) * 0.5
    
    plt.hlines(y=width_heights, xmin=left_ips, xmax=right_ips, color="green", linestyle="-", linewidth=2, label="FWHM Width")

    plt.title(f"1D Spatial Intensity Profile along Center Line (ROI) | Tilt: {tilt_angle_deg:.2f} deg")
    plt.xlabel("Pixel Coordinate (Y-axis)")
    plt.ylabel("Grayscale Intensity")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")

    check_path_exists(output_path)
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()

def generate_hough_debug_plots(image_path, raw_data, output_dir):
    base_name = Path(image_path).stem
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return

    mid_x = raw_data["extracted"]["mid_x"]
    roi_bgr = img_bgr[:, mid_x:].copy()
    
    edges = raw_data["hough_data"]["edges"]
    lines = raw_data["hough_data"]["lines"]
    angles_deg = raw_data["hough_data"]["angles_deg"]

    canny_path = output_dir / f"{base_name}_01_canny_edges.png"
    imwrite_check(str(canny_path), edges)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(roi_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    overlay_path = output_dir / f"{base_name}_02_hough_lines_overlay.png"
    imwrite_check(str(overlay_path), roi_bgr)

    if angles_deg:
        plt.figure(figsize=(8, 5))
        plt.hist(angles_deg, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        mean_angle = np.mean(angles_deg)
        median_angle = np.median(angles_deg)
        plt.axvline(mean_angle, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_angle:.3f} deg')
        plt.axvline(median_angle, color='green', linestyle='solid', linewidth=2, label=f'Median: {median_angle:.3f} deg')
        plt.title('Hough Lines Angle Distribution')
        plt.xlabel('Tilt Angle (Degrees)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.legend()
        
        hist_path = output_dir / f"{base_name}_03_angle_histogram.png"
        check_path_exists(hist_path)
        plt.savefig(str(hist_path), dpi=300, bbox_inches='tight')
        plt.close()

# ==========================================
# 批次處理函式
# ==========================================

# def run_batch(image_folder, output_filename="Master_Measurement_Report.xlsx"):
#     """批次處理：只做計算與匯總數據，不產生除錯圖表。"""
#     processor = FWHMProcessor()
#     image_folder = Path(image_folder)
#     image_paths = sorted([p for p in image_folder.iterdir() if p.suffix.lower() in [".jpg", ".png", ".bmp"]])

#     all_results = []
#     for image_path in image_paths:
#         print(f"Processing {image_path}...")
#         try:
#             result = processor.process(image_path)
#             all_results.append(result["df"])
#         except Exception as err:
#             print(f"Error processing {image_path}: {err}")

#     if not all_results:
#         print("Warning: No data to export.")
#         return None

#     master_df = pd.concat(all_results, ignore_index=True)
#     output_path = Path.cwd() / "data" / "results" / output_filename
    
#     check_path_exists(output_path)
#     master_df.to_excel(str(output_path), index=False)
#     print(f"Master Data successfully exported to {output_path}")
#     return output_path


# ==========================================
# 主程式執行區塊
# ==========================================

if __name__ == "__main__":
    # 測試單張圖片並繪製除錯圖表的流程
    default_image = "./data/measure_01.jpg"
    image_path = Path(default_image)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path.cwd() / "data" / "fwhm_hough_result" / timestamp
    check_path_exists(output_dir)

    print("Running processor...")
    processor = FWHMProcessor()
    result = processor.process(default_image)

    print("Generating visual plots...")
    draw_results_image(image_path, result["raw_data"], output_dir / f"{result['base_name']}_result.png")
    plot_intensity_profile(result["raw_data"], output_dir / f"{result['base_name']}_intensity_profile.png")
    generate_hough_debug_plots(image_path, result["raw_data"], output_dir)

    print("Saving measurement report...")
    report_path = output_dir / f"{result['base_name']}_measurement_report.xlsx"
    check_path_exists(report_path)
    result["df"].to_excel(str(report_path), index=False)
    
    print(f"Single image test complete. Results saved in {output_dir}")