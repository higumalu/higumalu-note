---

title: pydicomRT｜"🛠️ Image Series Loader — 錯誤處理改進"
date: 2026-01-06 12:00:00
updated: 2026-01-06 12:00:00
tags: [pydicomRT, 開發日誌]
categories: [pydicomRT]
---

## 🛠️ Image Series Loader — 錯誤處理改進

**Commit:** `d03e995`  
**日期：** 2026-01-06

---

### 📋 變動內容

這次我改進了 `image_series_loader.py` 中 DICOM 檔案排序的錯誤處理。

**修正內容：**
```python
# 修正前
image_ds_list = sort_ds_list(image_ds_list)

# 修正後
try:
    image_ds_list = sort_ds_list(image_ds_list)
except Exception as e:
    raise ValueError(f"Error sorting DICOM files: {e}, check if the ImageOrientationPatient attribute is valid")
```

**改進說明：**

1. **更明確的錯誤訊息** — 當排序失敗時，會顯示更具體的錯誤原因，特別是指向 `ImageOrientationPatient` 屬性的有效性檢查。

2. **提升除錯效率** — 在處理大量 DICOM 檔案時，如果某些檔案的 `ImageOrientationPatient` 缺失或格式錯誤，開發者可以快速定位問題。

3. **統一的例外處理** — 將各種可能的排序錯誤包裝成 `ValueError`，讓呼叫端可以使用統一的方式處理。

### 📁 受影響檔案

```
src/pydicomrt/utils/image_series_loader.py
  - load_sorted_image_series 函數錯誤處理強化
```

---

### 💭 熊熊的開發心得

**Q1: `ImageOrientationPatient` 缺失或格式錯誤會導致什麼問題？**
> 這個屬性是影像空間定位的關鍵。如果它不見了或格式不對，切片排序會亂掉，3D 重建也會出問題。之前錯誤訊息只說「排序失敗」，根本不知道問題在哪，現在至少知道要去檢查這個屬性。

**Q2: 除了錯誤處理，有沒有預防措施可以確保檔案的正確性？**
> 理想的做法是加一個驗證函數，在讀取影像序列之前先檢查所有必要屬性是否完整。這塊我還在考慮怎麼設計比較好，現階段先從錯誤處理下手，讓問題浮現出來再說。

**Q3: 這個改進對於批量處理大量 DICOM 檔案時有什麼幫助？**
> 批量處理最怕的就是少數幾顆老鼠屎壞了一鍋粥。假設有 500 張切片，前面 499 張都正常，但第 300 張的方向參數有問題，沒有好的錯誤處理的話可能要跑到最後才知道出錯。現在可以直接從錯誤訊息看出哪張有問題，省下不少除錯時間。

---

📂 **查看完整差異：** [GitHub](https://github.com/higumalu/pydicomRT/commit/d03e995)