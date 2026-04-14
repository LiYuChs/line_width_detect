import cv2
import numpy as np
from PIL import Image
from aura_sr import AuraSR
import torch

MODEL_ID = "fal/AuraSR-v2"  

class AuraSRWrapper:
    """AuraSR super resolution upscaler wrapper"""
    
    def __init__(self, model_id=MODEL_ID):
        self.model_id = model_id
        self.model = None  # 初始化時「不」載入模型，節省顯示記憶體

    def load_model(self):
        """需要用到時才呼叫此方法將模型載入 VRAM"""
        if self.model is None:
            print(f"Loding AuraSR ({self.model_id}) ...")
            self.model = AuraSR.from_pretrained(self.model_id)

    def release_model(self):
        """release VRAM"""
        if self.model is not None:
            print(f"Releasing AuraSR model memory...")
            del self.model
            self.model = None
            torch.cuda.empty_cache()

    def enhance(self, img_bgr):
        """
        統一的呼叫介面：輸入 OpenCV BGR 影像，輸出放大 4 倍的 OpenCV BGR 影像
        """
        self.load_model()  # 確保模型已載入

        # 1. OpenCV (BGR) 轉為 PIL (RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # 2. 執行超解析放大 (預設固定放大 4 倍)
        upscaled_pil = self.model.upscale_4x(pil_img)

        # 3. PIL (RGB) 轉回 OpenCV (BGR)
        upscaled_rgb = np.array(upscaled_pil)
        upscaled_bgr = cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2BGR)

        # 為了相容你之前 RealESRGAN 的回傳格式 (img, None)
        return upscaled_bgr, None
    
    def enhance_gray(self, image_gray):
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        upscaled_bgr, _ = self.enhance(image_bgr)
        return cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2GRAY)
    
def main():
    sr_model = AuraSRWrapper()
    test_img_path = r"D:\line_width_detect_itri\data\measure_01.jpg"
    img_gray = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    upscaled_gray = sr_model.enhance_gray(img_gray)
    cv2.imwrite("upscaled_image.jpg", upscaled_gray)


if __name__ == "__main__":
    main()