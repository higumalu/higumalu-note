---

title: pydicomRT｜"BSpline Registration — 網格變形對位"
date: 2025-09-11 12:00:00
categories: [pydicomRT]
tags: [pydicomRT, 開發日誌]
---

## ✨ BSpline Registration — 網格變形對位

**Commit:** `c56a5cf`  
**日期：** 2025-09-11

---

### 📋 變動內容

Single commit 加入了一個 396 行的完整模組，包含：
- **Multi-resolution Registration** — 多解析度金字塔加速收斂
- **Various Optimizers** — L-BFGS、GD 等優化器
- **Various Metrics** — MI、MSD 等相似度度量
- **Isotropic Resampling** — 均勻採樣
- **Masking and Sampling Options** — 遮罩和採樣選項
- **Verbose Logging** — 方便除錯

---

### 💭 熊熊的開發心得

**Q1: BSpline Registration 和 Demons Registration 有什麼不同？**
> 兩者都是非剛性對位方法，但定位不太一樣。BSpline 更像是實驗過程中的產物，實際使用上我會以 Demons Registration 為主。

**Q2: 為什麼最終選擇以 Demons 為主，而不是 BSpline？**
> 主要是效能考量。BSpline Registration 的計算時間太長，在實際應用中不太可行。相較之下 Demons 的速度友好多了。

**Q3: 那 BSpline 這份 code 的意義是什麼？**
> 這是實驗過程中的產物。我把它留下來作為記錄和參考，但不建議在正式場景使用。


---

📂 **查看完整差異：** [GitHub](https://github.com/higumalu/pydicomRT/commit/c56a5cf)