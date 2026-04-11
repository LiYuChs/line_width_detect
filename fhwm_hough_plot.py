import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_hough_debug_plots(image_path, output_dir="data/results/debug"):
    # 確保輸出目錄存在
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    base_name = Path(image_path).stem

    # 讀取影像 (灰階與彩色)
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.imread(str(image_path))
    if img_gray is None or img_bgr is None:
        print(f"無法讀取影像: {image_path}")
        return

    # 設定右半部 ROI
    h, w = img_gray.shape
    mid_x = w // 2
    roi_gray = img_gray[:, mid_x:]
    roi_bgr = img_bgr[:, mid_x:].copy()

    # 1. 產生並儲存 Canny 邊緣圖
    edges = cv2.Canny(roi_gray, 50, 150, apertureSize=3)
    cv2.imwrite(str(out_path / f"{base_name}_01_canny_edges.png"), edges)

    # 2. 執行 Hough Transform 並疊加線段
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=100, 
        minLineLength=80, 
        maxLineGap=10
    )

    angles_deg = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 繪製偵測到的線段 (使用紅色，線寬 2)
            cv2.line(roi_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 計算角度 (過濾垂直線避免除以零)
            if x2 - x1 != 0:
                angle_rad = np.arctan((y2 - y1) / (x2 - x1))
                angles_deg.append(np.degrees(angle_rad))

    # 儲存疊加線段後的彩色影像
    cv2.imwrite(str(out_path / f"{base_name}_02_hough_lines_overlay.png"), roi_bgr)

    # 3. 繪製並儲存角度分佈直方圖
    if angles_deg:
        plt.figure(figsize=(8, 5))
        
        # 繪製直方圖
        counts, bins, patches = plt.hist(
            angles_deg, 
            bins=30, 
            color='skyblue', 
            edgecolor='black', 
            alpha=0.7
        )
        
        # 計算平均值與中位數
        mean_angle = np.mean(angles_deg)
        median_angle = np.median(angles_deg)
        
        # 在圖表上標示平均值與中位數
        plt.axvline(mean_angle, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_angle:.3f}°')
        plt.axvline(median_angle, color='green', linestyle='solid', linewidth=2, label=f'Median: {median_angle:.3f}°')
        
        plt.title('Hough Lines Angle Distribution')
        plt.xlabel('Tilt Angle (Degrees)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.legend()
        
        # 儲存直方圖
        plt.savefig(str(out_path / f"{base_name}_03_angle_histogram.png"), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"視覺化圖表已成功輸出至: {out_path.resolve()}")

if __name__ == "__main__":
    # 請替換為您實際的圖片路徑
    test_image = "./data/measure_01.jpg" 
    generate_hough_debug_plots(test_image)