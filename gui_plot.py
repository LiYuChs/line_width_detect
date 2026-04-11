import base64
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np


def fit_image_for_panel(image_bgr, max_w, max_h):
    h, w = image_bgr.shape[:2]
    scale = min(max_w / w, max_h / h)
    scale = min(scale, 1.0)
    out_w = max(1, int(w * scale))
    out_h = max(1, int(h * scale))
    return cv2.resize(image_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)


def bgr_to_tk_photo(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ok, encoded = cv2.imencode(".png", rgb)
    if not ok:
        raise RuntimeError("無法將影像轉換為 tkinter PhotoImage")
    b64 = base64.b64encode(encoded.tobytes())
    return tk.PhotoImage(data=b64)


def build_titled_panel(image_bgr, title, panel_w=800, panel_h=450, title_h=42):
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:, :] = (18, 18, 18)

    resized = fit_image_for_panel(image_bgr, panel_w - 20, panel_h - title_h - 20)
    rh, rw = resized.shape[:2]
    x0 = (panel_w - rw) // 2
    y0 = title_h + (panel_h - title_h - rh) // 2
    panel[y0:y0 + rh, x0:x0 + rw] = resized

    cv2.rectangle(panel, (0, 0), (panel_w - 1, title_h - 1), (50, 50, 50), -1)
    cv2.putText(panel, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)
    return panel


def save_result_grid(output_path, roi_img, upscaled_roi, sam_segmentation, morphology):
    panel_1 = build_titled_panel(roi_img, "ROI")
    panel_2 = build_titled_panel(upscaled_roi, "Upscaled ROI")
    panel_3 = build_titled_panel(sam_segmentation, "SAM Segmentation")
    panel_4 = build_titled_panel(morphology, "After Morphology")

    top_row = cv2.hconcat([panel_1, panel_2])
    bottom_row = cv2.hconcat([panel_3, panel_4])
    grid = cv2.vconcat([top_row, bottom_row])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)


def save_individual_results(base_dir, image_name, roi_img, upscaled_roi, sam_segmentation, morphology):
    targets = {
        "roi": roi_img,
        "upscaled_roi": upscaled_roi,
        "sam_segmentation": sam_segmentation,
        "morphology": morphology,
    }

    saved_paths = {}
    for folder_name, image in targets.items():
        folder = base_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        out_path = folder / f"{image_name}_sam.jpg"
        cv2.imwrite(str(out_path), image)
        saved_paths[folder_name] = out_path

    return saved_paths


class SAMGuiApp:
    def __init__(self, root, processor):
        self.root = root
        self.processor = processor
        self.tk_images = {}

        self.root.title("SAM ROI Segmentation Viewer")
        self.root.geometry("1440x920")
        self.root.minsize(1200, 760)

        top = ttk.Frame(root, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        self.open_btn = ttk.Button(top, text="選擇影像並執行", command=self.on_open_image)
        self.open_btn.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="請選擇一張影像開始")
        self.status_label = ttk.Label(top, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=12)

        grid = ttk.Frame(root, padding=10)
        grid.pack(fill=tk.BOTH, expand=True)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

        self.panels = {}
        panel_specs = [
            ("roi", "ROI 區域", 0, 0),
            ("upscaled_roi", "放大 ROI 區域", 0, 1),
            ("sam_segmentation", "SAM segmentation", 1, 0),
            ("morphology", "形態學運算後 image", 1, 1),
        ]

        for key, title, row, col in panel_specs:
            frame = ttk.LabelFrame(grid, text=title, padding=8)
            frame.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            lbl = ttk.Label(frame)
            lbl.pack(fill=tk.BOTH, expand=True)
            self.panels[key] = lbl

    def _update_panel(self, key, image_bgr):
        display_img = fit_image_for_panel(image_bgr, max_w=640, max_h=360)
        tk_img = bgr_to_tk_photo(display_img)
        self.tk_images[key] = tk_img
        self.panels[key].configure(image=tk_img)

    def on_open_image(self):
        image_path = filedialog.askopenfilename(
            title="選擇影像",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )

        if not image_path:
            return

        try:
            self.status_var.set("執行 SAM 推論中，請稍候...")
            self.root.update_idletasks()

            output = self.processor.process_image(image_path)

            self._update_panel("roi", output["roi"])
            self._update_panel("upscaled_roi", output["upscaled_roi"])
            self._update_panel("sam_segmentation", output["sam_segmentation"])
            self._update_panel("morphology", output["morphology"])

            image_name = Path(image_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = Path("sam_result") / timestamp
            save_path = base_dir / f"{image_name}_sam.jpg"
            save_result_grid(
                save_path,
                roi_img=output["roi"],
                upscaled_roi=output["upscaled_roi"],
                sam_segmentation=output["sam_segmentation"],
                morphology=output["morphology"],
            )

            individual_paths = save_individual_results(
                base_dir,
                image_name=image_name,
                roi_img=output["roi"],
                upscaled_roi=output["upscaled_roi"],
                sam_segmentation=output["sam_segmentation"],
                morphology=output["morphology"],
            )

            self.status_var.set(f"完成: 已儲存至 {base_dir}")
            messagebox.showinfo(
                "完成",
                "已儲存結果圖:\n"
                f"時戳資料夾: {base_dir}\n"
                f"合成圖: {save_path}\n"
                f"ROI: {individual_paths['roi']}\n"
                f"放大 ROI: {individual_paths['upscaled_roi']}\n"
                f"SAM segmentation: {individual_paths['sam_segmentation']}\n"
                f"形態學後 image: {individual_paths['morphology']}",
            )

        except Exception as exc:
            self.status_var.set("處理失敗")
            messagebox.showerror("錯誤", str(exc))


def run_gui(processor):
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")

    app = SAMGuiApp(root, processor)
    _ = app
    root.mainloop()
