---

title: pydicomRT｜"🔧 BSpline — 位移場類型從 VectorFloat32 改為 VectorFloat64"
date: 2026-02-04 12:00:00
updated: 2026-02-04 12:00:00
tags: [pydicomRT, 開發日誌]
categories: [pydicomRT]
---

## 🔧 BSpline — 位移場類型從 VectorFloat32 改為 VectorFloat64

**Commit:** `3fce17c`  
**日期：** 2026-02-04

---

### 📋 變動內容

這次我將 BSpline 配準函數中的位移場（Displacement Field）類型從 `sitkVectorFloat32` 改為 `sitkVectorFloat64`。

**修正內容：**
```python
# 修正前
dvf_image = sitk.TransformToDisplacementField(
    output_transform,
    sitk.sitkVectorFloat32,  # 32位浮點數
    ...
)

# 修正後
dvf_image = sitk.TransformToDisplacementField(
    output_transform,
    sitk.sitkVectorFloat64,  # 64位浮點數
    ...
)
```

**改進說明：**

1. **提高精度** — Float64 提供雙倍精度，避免在多次變換或累積誤差時造成影像品質下降。

2. **保持一致性** — 大多數醫學影像使用 Float64 作為內部計算精度，升級後可避免不必要的精度轉換。

3. **更準確的變形映射** — 特別是對於需要精細控制的 BSpline 變形，高精度位移場能更準確地描述組織變形。

### 📁 受影響檔案

```
src/pydicomrt/reg/method/bspline.py
  - bspline_registration 函數位移場類型更新
```

---

### 💭 熊熊的開發心得

**Q1: Float32 和 Float64 的差異在實際臨床應用中明顯嗎？**
老實說，在大多數情況下可能感覺不出來。但如果涉及到多次變形疊加或非常精細的劑量計算時，累積誤差的影響就不可忽視了。醫學領域對精度要求比較高，既然 SimpleITK 原生支援 Float64，我覺得用下去比較安心。

**Q2: 這個改變會增加記憶體使用量嗎？影響有多大？**
會的，理論上記憶體用量會變成兩倍。不過實際影響還好——BSpline 本來就不是我主要的對位方法，平常主要還是用 Demons。再說，現代醫療影像的資料量那麼大，一張位移場多佔的記憶體相比之下不算什麼。

**Q3: 為什麼 BSpline 特別需要高精度？其他配準方法（如 Demons）呢？**
BSpline 的變形控制點網格很細緻，對細節的表達能力更強。如果位移場精度不夠，那些精細的變形細節就會被吃掉，BSpline 的優勢就浪費掉了。Demons 的話，因為它的變形機制不太一樣，對精度的敏感度沒那麼高，我目前還是維持 Float32。

---

📂 **查看完整差異：** [GitHub](https://github.com/higumalu/pydicomRT/commit/3fce17c)