import os
import cv2
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from fwhm_main import FWHMProcessor
from super_resolution.realesrgan import SuperResolutionUpscaler
from super_resolution.aurasr import AuraSRWrapper
# from super_resolution.coz import ChainOfZoomWrapper

from utils import check_path_exists, imwrite_check

def get_augmentation_effects(img_gray, random_augmentation):
    # 新增隨機條件模擬影像情況
    if random_augmentation:
        effects = []
        if random.random() > 0.5:
            effects.append("noise")
        if random.random() > 0.5:
            effects.append("blur")
        if random.random() > 0.5:
            effects.append("illumination")

        if not effects:
            effects.append(random.choice(["noise", "blur", "illumination"]))
            # print(f"隨機增強效果: {effects}")

        if "noise" in effects:
            # Gaussian noise (mean=0, sigma=25)
            mean=0
            sigma=25
            gauss = np.random.normal(mean, sigma, img_gray.shape).astype(np.float32)
            img_gray = cv2.add(img_gray.astype(np.float32), gauss).clip(0, 255).astype(np.uint8)

        if "blur" in effects:
            img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

        if "illumination" in effects:
            if random.random() > 0.5:
                # 增加亮度
                alpha, beta = 1.2, 50
            else:
                # 減少亮度
                alpha, beta = 0.5, 64
            img_gray = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=beta)

        aug_output_path = output_dir / "augmented"
        check_path_exists(aug_output_path)
        aug_output_path = aug_output_path / f"augmented_{rotate_deg}.jpg"
        imwrite_check(str(aug_output_path), img_gray)

    return img_gray

def rotate_image(img_gray, processor):
    h, w = img_gray.shape
    mid_x = w // 2

    # 2. 計算原始角度
    orig_data = processor._calculate_tilt_angle(img_gray, mid_x)
    orig_angle = np.degrees(orig_data["median_angle_rad"])
    # print(f"[原始影像] 偵測傾角: {orig_angle:.3f} 度")

    # 3. 執行影像旋轉 (使用 cv2.warpAffine)
    # OpenCV 的 getRotationMatrix2D 中，正角度代表「逆時針旋轉」
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotate_deg, 1.0)
    
    # 為了避免旋轉後線條超出畫面被裁切，必須重新計算旋轉後的畫布尺寸 (Bounding Box)
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    # 修正旋轉中心的偏移量
    M[0, 2] += new_w / 2 - center[0]
    M[1, 2] += new_h / 2 - center[1]

    # 進行旋轉。使用 borderValue=255 (白色) 填滿背景，避免產生黑邊干擾 Canny 邊緣偵測
    rotated_gray = cv2.warpAffine(img_gray, M, (new_w, new_h), borderValue=255)
    new_mid_x = new_w // 2

    return orig_angle, rotated_gray, new_mid_x, new_w, new_h, h, w, M

def super_resolution_enhance(sr_upscaler, rotated_gray):
    if sr_upscaler is None:
        sr_upscaler = SuperResolutionUpscaler()
    upscaled_gray = sr_upscaler.enhance_gray(rotated_gray)
    sr_output_path = output_dir / "sr_enhanced"
    check_path_exists(sr_output_path)
    sr_output_path = sr_output_path / f"sr_enhanced_{rotate_deg}.jpg"
    imwrite_check(str(sr_output_path), upscaled_gray)
    return upscaled_gray       

def verify_hough_rotation(image_path, output_dir, rotate_deg=25.0, random_augmentation=False, use_super_resolution=False, sr_upscaler=None, process_order=False):
    print(f"--- 啟動旋轉驗證測試 (人為旋轉角度: {rotate_deg} 度) ---")

    # 1. 讀取原圖與初始化
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"找不到測試影像: {image_path}")
    
    img_gray = get_augmentation_effects(img_gray, random_augmentation)

    processor = FWHMProcessor()

    if process_order == 0:
        # 先旋轉再超解析
        orig_angle, rotated_gray, new_mid_x, new_w, new_h, h, w, M = rotate_image(img_gray, processor)

        if use_super_resolution:
            upscaled_gray = super_resolution_enhance(sr_upscaler, rotated_gray)
    else:
        # 先超解析再旋轉
        if use_super_resolution:
            upscaled_gray = super_resolution_enhance(sr_upscaler, img_gray)

        orig_angle, rotated_gray, new_mid_x, new_w, new_h, h, w, M = rotate_image(img_gray, processor)

    rot_data = processor._calculate_tilt_angle(upscaled_gray, new_mid_x * 4) # 因為放大 4 倍，所以 mid_x 也要乘以 4
    rot_angle = 0.0 - np.degrees(rot_data["median_angle_rad"])

    # print(f"[旋轉影像] 偵測傾角: {rot_angle:.3f} 度")


    # 5. 驗證差異
    # 因為原圖可能有微小初始傾角，實際測量到的總變化量應為兩者相減的絕對值
    angle_diff = abs(rot_angle + orig_angle)
    error = abs(angle_diff - rotate_deg)
    
    # print("-" * 50)
    # print(f"實際測量轉動差值: {angle_diff:.3f} 度")
    # print(f"與目標 ({rotate_deg}度) 誤差: {error:.3f} 度")
    # print("-" * 50)

    # --- 6. 視覺化旋轉角度 (旋轉邊框與角度扇形) ---
    # 將灰階圖轉為彩色以繪製彩色標記
    viz_img = cv2.cvtColor(rotated_gray, cv2.COLOR_GRAY2BGR)
    new_center = (new_w // 2, new_h // 2)

    # A. 繪製原始影像的旋轉邊框
    # 計算原圖的四個角落經過 M 轉換後的新座標
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ]).T
    transformed_corners = np.dot(M, corners).T
    pts = transformed_corners.astype(np.int32)
    # 畫出綠色外框表示原圖範圍
    cv2.polylines(viz_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # B. 繪製水平參考線 (0度) 與旋轉後的基準線
    line_length = min(new_w, new_h) // 3
    # 水平參考線 (藍色)
    cv2.line(viz_img, new_center, (new_center[0] + line_length, new_center[1]), (255, 0, 0), 2, cv2.LINE_AA)

    # 旋轉後的基準線 (紅色)
    rad = np.radians(-rotate_deg) # OpenCV Y軸向下，加上負號來修正畫布上的視覺方向
    end_x = int(new_center[0] + line_length * np.cos(rad))
    end_y = int(new_center[1] + line_length * np.sin(rad))
    cv2.line(viz_img, new_center, (end_x, end_y), (0, 0, 255), 2, cv2.LINE_AA)

    # C. 繪製半透明角度扇形
    overlay = viz_img.copy()
    radius = line_length // 2
    
    start_angle = 0
    end_angle = -rotate_deg
    # cv2.ellipse 繪製時要求 start_angle 需小於 end_angle
    if end_angle < start_angle:
        start_angle, end_angle = end_angle, start_angle

    # 畫出橘色實心扇形
    cv2.ellipse(overlay, new_center, (radius, radius), 0, start_angle, end_angle, (0, 165, 255), -1)
    # 疊加出半透明效果
    cv2.addWeighted(overlay, 0.4, viz_img, 0.6, 0, viz_img)

    # D. 加上文字標籤
    text_rad = np.radians(-rotate_deg / 2)
    text_x = int(new_center[0] + (radius + 20) * np.cos(text_rad))
    text_y = int(new_center[1] + (radius + 20) * np.sin(text_rad))
    cv2.putText(viz_img, f"{rotate_deg} deg", (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # 輸出視覺化圖檔供確認
    output_path = output_dir / "rotation"
    check_path_exists(output_path)
    output_path = output_path / f"rotation_viz_{rotate_deg}.jpg"
    imwrite_check(str(output_path), viz_img)

    return {
        "original_angle": orig_angle,
        "rotated_angle": rot_angle,
        "angle_diff": angle_diff,
        "error": error
    }

def plot_rotation_error(error_records, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(error_records, marker='o')
    plt.title("Rotation Angle Error vs. Target Angle")
    plt.xlabel("Target Rotation (degrees)")
    plt.ylabel("Error (degrees)")
    plt.xticks(range(0, 91, 10))
    plt.grid()
    plt.savefig(output_dir / "rotation_error_plot.png")
    plt.close()

def txt_writer(file_path, content):
    with open(file_path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_image = "./data/measure_01.jpg"

    output_dir = Path.cwd() / "data" / "results" / "hough_rotation" / timestamp
    check_path_exists(output_dir)
    

    # verify_hough_rotation(test_image, output_dir, rotate_deg=45.0, random_augmentation=True)
    
    # Low-VRAM mode: tile-based inference and half precision reduce peak GPU memory.
    # sr_upscaler = SuperResolutionUpscaler(tile=200, tile_pad=10, half=True)
    sr_upscaler = AuraSRWrapper()
    # sr_upscaler = ChainOfZoomWrapper(target_scale=4)

    error_records = []
    for rotate_deg in range(0, 2):
        result = verify_hough_rotation(
            test_image,
            output_dir,
            rotate_deg=rotate_deg,
            random_augmentation=True,
            use_super_resolution=True,
            sr_upscaler=sr_upscaler,
            process_order=True
        )
        error_records.append(result["error"])
        # AuraSRWrapper.release_model(sr_upscaler)

    sr_upscaler.release_model()

    plot_rotation_error(error_records, output_dir)

    # 輸出統計結果到文字檔
    stats_content = f"--- 旋轉角度誤差統計 ---\n"
    stats_content += f"最大誤差： {max(error_records):.3f} 度\n"
    stats_content += f"最小誤差： {min(error_records):.3f} 度\n"
    stats_content += f"平均誤差： {sum(error_records)/len(error_records):.3f} 度\n"

    txt_writer(output_dir / "rotation_stats.txt", stats_content)