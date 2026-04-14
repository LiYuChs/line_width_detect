# line_width_detect_itri

本專案用於線寬量測與流程驗證，整合了多種方法：
- FWHM + Hough 傾角補償
- SAM 直接分割
- Optimized SAM（ROI + 上採樣 + morphology）
- SR + SAM（RealESRGAN）
- AuraSR 包裝與旋轉角度驗證測試

## 1. 目前程式碼進度

### 已完成
- 核心 FWHM 流程已可運作（含傾角估計、補償、量測報表輸出）
- SAM 與 Optimized SAM 的單張圖流程可執行
- SR + SAM（RealESRGAN）流程可執行，並可輸出分割遮罩與報表
- Hough rotation 驗證腳本可執行，並可輸出誤差曲線與統計
- AuraSR 包裝器已具備彩色/灰階增強方法與 VRAM 釋放接口

### 進行中 / 待整理
- 主入口 `main.py` 尚未與目前各 Processor 介面完全對齊（目前 Processor 主要使用 `process()`）
- 目前較偏研究腳本型態，尚未完成統一 CLI 與正式測試流程
- README（本檔）為現況整理版，後續可再補上範例結果圖

## 2. 專案結構（重點）

- `fwhm_main.py`：FWHM 主流程與 Hough debug 圖
- `sam_main.py`：SAM 直接分割流程
- `optimized_sam_main.py`：Optimized SAM
- `sr_sam_main.py`：RealESRGAN + SAM
- `super_resolution/realesrgan.py`：RealESRGAN 共用 wrapper
- `super_resolution/aurasr.py`：AuraSR wrapper（測試/替代 SR）
- `tests/test_hough_rotation.py`：旋轉角度驗證（可選擇是否套用 SR）
- `utils.py`：影像輸出與資料夾建立工具
- `data/`：各流程輸出與歷史結果

## 3. 環境需求

建議 Python 3.8+（目前實際使用為 conda 環境）。

主要套件：
- opencv-python
- matplotlib
- numpy
- pandas
- scipy
- torch
- segment-anything
- basicsr
- realesrgan
- pillow
- aura-sr

可先安裝基本版（不含 AuraSR）：

```bash
pip install opencv-python matplotlib numpy pandas scipy torch pillow
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install basicsr realesrgan
```

若要啟用 AuraSR，另安裝：

```bash
pip install aura-sr
```

## 4. 權重檔案

請確認以下檔案存在：
- `pretrained_weights/sam_vit_b_01ec64.pth`
- `pretrained_weights/RealESRGAN_x4plus.pth`
- （選用）`pretrained_weights/sam_vit_l_0b3195.pth`

## 5. 目前可用執行方式

### FWHM

```bash
python fwhm_main.py
```

輸出範例：
- `data/fwhm_hough_result/<timestamp>/...`

### SAM

```bash
python sam_main.py
```

輸出範例：
- `data/sam_result/...`

### Optimized SAM

```bash
python optimized_sam_main.py
```

輸出範例：
- `data/opt_sam_result/...`

### SR + SAM（RealESRGAN）

```bash
python sr_sam_main.py
```

輸出範例：
- `data/sr_sam_result/<timestamp>/...`

### Hough Rotation 驗證

```bash
python tests/test_hough_rotation.py
```

輸出範例：
- `data/results/hough_rotation/<timestamp>/rotation_error_plot.png`
- `data/results/hough_rotation/<timestamp>/rotation_stats.txt`

## 6. 已知問題與注意事項

1. `main.py` 目前仍有介面整合問題
- 目前 `main.py` 使用 `processor.run(...)`，但現有 Processor 類別主要提供 `process(...)`。
- 建議短期先直接執行各 `*_main.py` 腳本。

2. `main.py` 匯入名稱大小寫問題
- `from SAM_main import SAMProcessor` 可能在大小寫敏感系統失敗。
- 建議改為 `from sam_main import SAMProcessor`。

3. AuraSR 相容性
- `aura_sr` 套件在部分 Python 版本可能有型別註解相容問題（例如匯入階段錯誤）。
- 在aura-sr.py中，最上面加上 from __future__ import annotations。
- 若遇到 AuraSR 匯入錯誤，可先切回 `SuperResolutionUpscaler`（RealESRGAN）路徑。

4. 測試現況
- `tests/` 目前以流程驗證腳本為主，尚未建立完整 pytest 自動化測試結構。

## 7. 建議下一步

- 統一 Processor 介面（`run()` / `process()`）並修正 `main.py`
- 將 `tests/test_hough_rotation.py` 改為 pytest 形式（可參數化旋轉角）
- 補齊 `requirements.txt` 或 `environment.yml`
- 在 `report/` 補上固定格式的量測報告模板與圖表

---
