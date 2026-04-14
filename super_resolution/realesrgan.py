from pathlib import Path
import warnings
import gc

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_REALESRGAN_CHECKPOINT_PATH = "./pretrained_weights/RealESRGAN_x4plus.pth"
DEFAULT_UPSCALE_FACTOR = 4


class SuperResolutionUpscaler:
    """Shared Real-ESRGAN wrapper for image enhancement pipelines."""

    def __init__(
        self,
        checkpoint_path=DEFAULT_REALESRGAN_CHECKPOINT_PATH,
        upscale_factor=DEFAULT_UPSCALE_FACTOR,
        device=DEVICE,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=None,
    ):
        self.checkpoint_path = checkpoint_path
        self.upscale_factor = upscale_factor
        self.device = device
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        if half is None:
            self.half = str(self.device).startswith("cuda")
        else:
            self.half = half

        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"RealESRGAN checkpoint not found: {self.checkpoint_path}")

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=self.upscale_factor,
        )
        # realesrgan currently calls torch.load without weights_only, which emits a FutureWarning on newer torch.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*torch\.load.*weights_only=False.*",
                category=FutureWarning,
            )
            self.upsampler = RealESRGANer(
                scale=self.upscale_factor,
                model_path=self.checkpoint_path,
                model=model,
                tile=self.tile,
                tile_pad=self.tile_pad,
                pre_pad=self.pre_pad,
                half=self.half,
                device=self.device,
            )

    def enhance(self, image_bgr):
        upscaled_img, _ = self.upsampler.enhance(image_bgr, outscale=self.upscale_factor)
        return upscaled_img

    def enhance_gray(self, image_gray):
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        upscaled_bgr = self.enhance(image_bgr)
        return cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def clear_cuda_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release(self):
        if hasattr(self, "upsampler"):
            del self.upsampler
        gc.collect()
        self.clear_cuda_cache()
