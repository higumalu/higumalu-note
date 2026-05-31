---

title: pydicomRT｜"Registration 模組進場"
date: 2025-08-18 12:00:00
categories: [pydicomRT]
tags: [pydicomRT, 開發日誌]
---

## ✨ Registration 模組進場

**Commit:** `6cc3899`  
**日期：** 2025-08-18

---

### 📋 變動內容

經過兩個月的沈澱，我的 Registration 模組終於進場！

這次我加入了：
- **Rigid Registration** — 剛性對位（旋轉 + 平移）
- **Demons Registration** — 基於光流法的非剛性對位
- **Soft Mask 支援** — 可以針對特定 ROI 做對位
- **Image Resampling** — 對位後的圖像重採樣工具

這次改動 8 個檔案、+854 / -146 行，是一個相當大的功能更新。

### 📁 受影響檔案

```
8 files changed, 854 insertions(+), 146 deletions(-)
src/pydicomrt/reg/method/
src/pydicomrt/utils/sitk_transform.py
```

---

### 💭 熊熊的開發心得

**Q1: 在一開始規劃套件時就考慮進去了？**
> 沒錯，Registration 模組是從一開始就在我的藍圖裡的。RT 流程中離不開影像對位，所以規劃之初就把它列為必要的擴展方向。

**Q2: 實作 Registration 模組之前，有先研究過哪些現成方案嗎？**
> 一開始我想用 scipy + numpy 自己刻，但後來發現 Resample 的部分會變得過於複雜。評估過後我決定採用 SimpleITK 作為 Registration 模組的主要方法——成熟、穩定、文件完整，省去大量自己造輪子的力氣。

**Q3: 這部分跟臨床流程有關？Rigid 和 Deformable 兩種對位方式怎麼來的？**
> 對，完全是從臨床流程來的。Rigid（剛性對位）貼近 RT 儀器的實際操作邏輯——患者體位固定後，透過平移和旋轉就能完成對位，不需要複雜的變形。Deformable（變形對位）則是另一個臨床需求：用於將 CT sim 的 ROI 變形到每日治療的 CT 上，應對患者每天擺位的些微差異。


---

📂 **查看完整差異：** [GitHub](https://github.com/higumalu/pydicomRT/commit/6cc3899)