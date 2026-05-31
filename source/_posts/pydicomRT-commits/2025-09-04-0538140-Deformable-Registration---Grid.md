---

title: pydicomRT｜"Deformable Registration — Grid 屬性實作"
date: 2025-09-04 12:00:00
categories: [pydicomRT]
tags: [pydicomRT, 開發日誌]
---

## ✨ Deformable Registration — Grid 屬性實作

**Commit:** `0538140`  
**日期：** 2025-09-04

---

### 📋 變動內容

這次我實作了 Deformable Registration 的 Grid 屬性，為未來產生 DeformableRegistration DICOM 打下基礎。

我加入了：
- **DeformableSpatialRegistrationBuilder** — 簡化 constructor，專門用於非剛性對位的 Builder
- **Grid 屬性** — `GridDimensions`、`GridResolution`、`VectorGridData`、`ImageOrientationPatient`
- **Deformation Matrix Sequences** — `PreDeformationMatrixRegistrationSequence` 與 `PostDeformationMatrixRegistrationSequence`

不過這部分目前仍在研究階段，實際的 DICOM 產生功能尚未完成。

### 📁 受影響檔案

```
2 files changed, 43 insertions(+), 6 deletions(-)
src/pydicomrt/reg/builder.py
src/pydicomrt/reg/parser.py
```

---

### 💭 熊熊的開發心得

**Q1: 這次把 Builder constructor 大幅簡化了，把參數從 Init 移到 `add_deformable_registration` 方法裡。當時為什麼這樣改？**
> 主要是 API 設計的考量。Constructor 放太多參數會讓使用起來很不直覺，後來我改成在要加入對位資料時再傳入，流程更清晰。

**Q2: 實作這串 Sequence 的過程中，最麻煩的部分是什麼？**
> 說實話，這部分還沒完成。這次 commit 其實是朝著「產生 DeformableRegistration DICOM」這個目標前進，但實作過程中我發現底層細節還需要更多研究，目前處於研究階段，尚未真正完成。

**Q3: 這次改動的意義是什麼？**
> 這次雖然還沒辦法實際產生完整的 DeformableRegistration DICOM，但已經把核心的資料結構墊好了。Grid 屬性和 Sequence 的框架先建立，未來研究完成後就能直接往下接。


---

📂 **查看完整差異：** [GitHub](https://github.com/higumalu/pydicomRT/commit/0538140)