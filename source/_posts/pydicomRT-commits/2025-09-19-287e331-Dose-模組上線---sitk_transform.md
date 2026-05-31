---

title: pydicomRT｜"Dose 模組上線 — sitk_transform"
date: 2025-09-19 12:00:00
categories: [pydicomRT]
tags: [pydicomRT, 開發日誌]
---

## ✨ Dose 模組上線 — sitk_transform

**Commit:** `287e331`  
**日期：** 2025-09-19

---

### 📋 變動內容

Dose（劑量）處理是 pydicomRT 的第三個主要模組。

這次我加入的 `sitk_transform` 模組解決了一個核心問題：
- **DICOM Dose Array ↔ SimpleITK Image** — 讓劑量資料可以在 DICOM 格式和 SimpleITK 格式之間自由轉換
- **方向處理** — Dose 有特殊的 Direction 需要處理

`sitk_transform` 的定位是「DICOM 資料與 SimpleITK 資料之間的橋樑」，讓各類資料能夠互通。

### 📁 受影響檔案

```
1 file changed, 39 insertions(+)
src/pydicomrt/dose/sitk_transform.py
```

---

### 💭 熊熊的開發心得

**Q1: Dose 在整個 RT 流程中扮演什麼角色？**
> Dose 是 RT 資料中相當重要的一個區塊，源自於 RT Plan 的資料，主要用於紀錄治療劑量相關的資訊。實務上我會搭配 Registration 使用，讓放射物理師更好去評估每日治療劑量是否符合一開始規劃的劑量。

**Q2: sitk_transform 解決了什麼問題？**
> 一開始寫這個方法是為了讓各類資料能夠更好地在 DICOM 格式和 SimpleITK 格式之間互通。SimpleITK 有很強大的影像處理工具，但 DICOM 的資料結構畢竟不同，需要一個乾淨的轉換層。

**Q3: Dose 資料處理的難點在哪裡？**
> 轉換到患者座標空間是一個難題，另一部分則是要如何把 Dose 資料很好地綁定到影像資料的空間上。兩邊座標對不上，後續的劑量評估就會失準。


---

📂 **查看完整差異：** [GitHub](https://github.com/higumalu/pydicomRT/commit/287e331)