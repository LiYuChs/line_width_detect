import cv2
import numpy as np
from PIL import Image
import io
import base64
import requests
import fal_client

class ChainOfZoomWrapper:
    """Chain-of-Zoom (CoZ) 超解析模型的純 API 封裝類別"""
    
    def __init__(self, target_scale=4):
        self.target_scale = target_scale

    def load_model(self):
        """API 模式不需要在本地載入模型，此處保留作為介面統一"""
        pass

    def release_model(self):
        """API 模式不需要釋放 VRAM"""
        pass

    def enhance(self, img_bgr, center_x=0.5, center_y=0.5):
        """
        統一的影像處理介面
        Args:
            img_bgr: OpenCV BGR 影像陣列
            center_x, center_y: 局部放大的正規化中心座標 (0.0 ~ 1.0)。
                                預設 0.5, 0.5 為從正中央放大。
        """
        print(f"正在將圖片上傳至 fal-ai 進行 Chain-of-Zoom {self.target_scale}x 放大...")
        
        # 1. OpenCV (BGR) 轉為 PIL (RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # 2. 將圖片轉為 Base64 Data URI (讓 API 可以直接吃本機圖片)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{img_str}"

        # 3. 呼叫 fal-client 執行 CoZ
        try:
            result = fal_client.subscribe(
                "fal-ai/chain-of-zoom",
                arguments={
                    "image_url": data_url,
                    "scale": self.target_scale,
                    "center_x": center_x, # CoZ 允許你指定放大的焦點座標
                    "center_y": center_y
                },
            )
        except Exception as e:
            raise RuntimeError(f"API 呼叫失敗，請確認已設定環境變數 FAL_KEY。錯誤細節: {e}")

        # 4. 下載雲端算好的圖片
        print("雲端運算完成，正在下載結果...")
        response = requests.get(result['image']['url'])
        upscaled_pil = Image.open(io.BytesIO(response.content)).convert("RGB")

        # 5. PIL (RGB) 轉回 OpenCV (BGR)
        upscaled_rgb = np.array(upscaled_pil)
        upscaled_bgr = cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2BGR)

        # 回傳格式保持與 RealESRGAN/AuraSR 一致
        return upscaled_bgr, None
    
    def enhance_gray(self, image_gray, center_x=0.5, center_y=0.5):
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        upscaled_bgr, _ = self.enhance(image_bgr, center_x, center_y)
        return cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2GRAY)