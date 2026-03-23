import cv2
import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from pathlib import Path

class ImageAnalyzer:
    """Handles image processing and pixel-level measurements."""
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.h, self.w = 0, 0
        self.mid_x = 0
        self.roi_start, self.roi_end = 350, 550
        
    def extract_fwhm_widths(self):
        """Extracts peak positions and FWHM pixel widths from the center column."""
        if self.img is None:
            raise FileNotFoundError("Error: Could not load the image.")
            
        self.h, self.w = self.img.shape
        self.mid_x = self.w // 2 # 350
        col_intensities = self.img[:, self.mid_x]
        section = col_intensities[self.roi_start:self.roi_end]
        
        # Find peaks and calculate FWHM
        peaks, _ = find_peaks(section, height=40, prominence=15, distance=5)
        pixel_widths, _, left_ips, right_ips = peak_widths(section, peaks, rel_height=0.5)
        
        # Return absolute Y coordinates and sub-pixel widths
        abs_peaks = peaks + self.roi_start
        abs_left_ips = left_ips + self.roi_start
        abs_right_ips = right_ips + self.roi_start
        
        return abs_peaks, pixel_widths, abs_left_ips, abs_right_ips

class Calibrator:
    """Handles physical scale calibration."""
    @staticmethod
    def calculate_scale(pixel_widths, known_physical_widths):
        """Calculates average scale (mm/pixel) using known references."""
        scales = []
        for i in range(len(known_physical_widths)):
            scale = known_physical_widths[i] / pixel_widths[i]
            scales.append(scale)
        average_scale = sum(scales) / len(scales)
        return average_scale

class DataFormatter:
    """Handles visualization and data export (Excel)."""
    @staticmethod
    def draw_results(image_path, peaks, left_ips, right_ips, physical_widths_mm, output_filename):
        """Draws measurement markers and physical text on the image."""
        img_bgr = cv2.imread(image_path) 
        if img_bgr is None:
            return
            
        h, w, _ = img_bgr.shape
        mid_x = w // 2
        
        # Draw center line
        cv2.line(img_bgr, (mid_x, 0), (mid_x, h - 1), (0, 0, 255), 1)
        
        for i in range(len(peaks)):
            y_peak = int(round(peaks[i]))
            y_left = int(round(left_ips[i]))
            y_right = int(round(right_ips[i]))
            
            # Draw peak marker and FWHM bar
            cv2.line(img_bgr, (mid_x - 10, y_peak), (mid_x + 10, y_peak), (0, 255, 0), 1)
            cv2.line(img_bgr, (mid_x + 25, y_left), (mid_x + 25, y_right), (0, 255, 0), 2)
            
            # Put text (Physical width in mm)
            text = f"L{i+1}: {physical_widths_mm[i]:.3f}mm"
            cv2.putText(img_bgr, text, (mid_x + 35, y_peak + 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                        
        output_path = Path.cwd() / 'data' / 'results' / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img_bgr)
        print(f"Visualization saved to {output_path}")

    @staticmethod
    def export_batch_to_excel(all_data_frames, output_filename):
        """Merges all individual DataFrames and exports to a single Master Excel file."""
        if not all_data_frames:
            print("Warning: No data to export.")
            return
            
        # Concatenate all individual DataFrames into one master DataFrame
        master_df = pd.concat(all_data_frames, ignore_index=True)
        
        output_path = Path.cwd() / 'data' / 'results' / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        master_df.to_excel(str(output_path), index=False)
        print(f"Master Data successfully exported to {output_path}")

class SingleProcessor:
    """Handles the entire process for a single image and returns its data."""
    def __init__(self, image_path):
        self.image_path = image_path
        self.analyzer = ImageAnalyzer(image_path)
        
    def run(self):
        """Processes the image and returns a pandas DataFrame with the results."""
        peaks, pixel_widths, left_ips, right_ips = self.analyzer.extract_fwhm_widths()
        
        # Calibration using the first 4 lines (known ground truths)
        known_ground_truths_mm = [0.015, 0.016, 0.015, 0.016]
        scale_mm_per_px = Calibrator.calculate_scale(pixel_widths[:4], known_ground_truths_mm)
        
        # Apply scale to all lines
        all_physical_widths_mm = [round(pw * scale_mm_per_px, 3) for pw in pixel_widths]
        
        # Extract the base name for the image output and DataFrame identifier
        base_name = Path(self.image_path).stem
        img_output_name = f"{base_name}_result.png"
        
        # Visualization
        DataFormatter.draw_results(
            self.image_path, peaks, left_ips, right_ips, all_physical_widths_mm, img_output_name
        )
        
        # Prepare data dictionary, including the image name for the master table
        data = {
            "Image Name": [base_name] * len(peaks),
            "Line ID": [f"Line {i+1}" for i in range(len(peaks))],
            "Y Coordinate (Pixel)": peaks,
            "Pixel Width (FWHM)": pixel_widths,
            "Physical Width (mm)": all_physical_widths_mm,
            "Applied Scale (mm/px)": [scale_mm_per_px] * len(peaks)
        }
        
        # Return as DataFrame for easy batch concatenation
        return pd.DataFrame(data)

class BatchProcessor:
    """Handles batch processing of multiple images and creates a master report."""
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
        
    def run(self):
        all_results = [] # List to store DataFrames from each image
        
        for img_path in self.image_paths:
            print(f"Processing {img_path}...")
            try:
                processor = SingleProcessor(img_path)
                df_result = processor.run()
                all_results.append(df_result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        # After loop finishes, export the merged data
        master_excel_name = "Master_Measurement_Report.xlsx"
        DataFormatter.export_batch_to_excel(all_results, master_excel_name)

# ==========================================
# Main Execution Pipeline
# ==========================================
if __name__ == "__main__":
    img_path = './data/measure_01.jpg' # Use the original image for clean drawing
    processor = SingleProcessor(img_path)
    processor.run()

    # image_folder = './data' # Folder containing all images to process
    # processor = BatchProcessor(image_folder)
    # processor.run()