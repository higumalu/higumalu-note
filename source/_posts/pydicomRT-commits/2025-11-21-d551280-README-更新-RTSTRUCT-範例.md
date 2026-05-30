---
title: "📝 README 更新 — RTSTRUCT 轉換範例加入 Affine 映射"
date: 2025-11-21 12:00:00
updated: 2025-11-21 12:00:00
tags: [pydicomRT, 開發日誌]
categories: [pydicomRT]
---

## 📝 README 更新 — RTSTRUCT 轉換範例加入 Affine 映射

**Commit:** `d551280`  
**日期：** 2025-11-21

---

### 📋 變動內容

這次我更新了 README 文件中 RTSTRUCT 转换为 3D Mask 的使用範例：

1. **新增 `calc_image_series_affine_mapping` 函數** — 這個函數用於計算影像序列的仿射映射（affine mapping）和遮罩體積形狀（mask volume shape）。

2. **更新 `rtstruct_to_mask_dict` 調用方式** — 從原本直接傳入 `ds_list`，改為傳入預先計算好的 `affine_mapping` 和 `mask_volume_shape`，提升 3D 遮罩生成的效率和準確性。

這個改變讓使用者能夠更清楚地理解如何正確處理 DICOM 影像的几何資訊，特別是在處理具有傾斜掃描（gantry tilt）的 CT 影像時，仿射映射的正確性至關重要。

### 📁 受影響檔案

```
README.md
  - 更新 RTSTRUCT 轉換範例（中英文）
README_zh.md
  - 更新 RTSTRUCT 轉換範例（繁體中文）
```

---

### 💭 熊熊的開發心得

**Q1: `calc_image_series_affine_mapping` 計算的是什麼？和 DICOM 的哪個屬性有關？**
> 它計算的是 ImagePositionPatient 和 ImageOrientationPatient 這兩個 DICOM 屬性的組合，用來描述影像在患者空間中的位置和方向。簡單說就是告訴我「這張切片在世界座標的哪裡、指向哪個方向」。

**Q2: 為什麼要分開計算 affine mapping 再傳入，而不是在函數內部處理？**
> 兩個原因。一是讓使用者更清楚整個流程，知道這些幾何資訊是從哪裡來的；二是避免重複計算——如果一次要轉很多張結構，預先算好 affine mapping 會更有效率。

**Q3: 傾斜掃描（gantry tilt）的問題在這裡是怎麼處理的？**
> 這個問題我之前也被坑過。帶有 gantry tilt 的 CT 影像沒有辦法用簡單的等距切片假設，必須靠正確的 affine mapping 才能正確重建 3D 幾何。`calc_image_series_affine_mapping` 會根據每張切片的實際位置和方向參數，分別計算對應的仿射變換矩陣。

---

📂 **查看完整差異：** [GitHub](https://github.com/higumalu/pydicomRT/commit/d551280)