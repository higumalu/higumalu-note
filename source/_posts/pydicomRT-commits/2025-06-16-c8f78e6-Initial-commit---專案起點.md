---
title: "Initial commit — 專案起點"
date: 2025-06-16 12:00:00
categories: [pydicomRT]
tags: [pydicomRT, 開發日誌]
---

## 🏁 Initial commit — 專案起點

**Commit:** `c8f78e6`  
**日期：** 2025-06-16

---

### 📋 變動內容

這是 pydicomRT 的第一個 commit，我建立了專案的基本結構：`.gitignore`、`LICENSE`、`README.md`。

我決定從零開始建立一個專案，選擇正確的工具和架構是很重要的第一步。

### 📁 受影響檔案

```
3 files changed, 216 insertions(+)
.gitignore
LICENSE
README.md
```

---

### 💭 熊熊的開發心得

**Q1: 當時是什麼契機想要建立這個專案？**
> 我把醫學影像操作變簡單——這是最初的念頭。要把 DICOM 處理成可以實際操作的物件，往往得先深入研究儀器的成像原理與資料型態，其中 RT（放射治療）的需求特別複雜。我希望打造一個簡單的套件，讓研究人員用幾行 code 就能操作這些複雜資料，不必自己踩坑。

**Q2: 為什麼選擇這個專案結構？（src/ + example/ + test/）**
> 主要是套件開發的慣例，沒有特別設計。src/ 內的架構是我希望根據 RT 每個流程的資料型態來分門別類：影像相關、RTSTRUCT 相關、DOSE 相關、影像配準相關——每個子領域一套對應的 function。

**Q3: pyproject.toml 的設計有參考什麼嗎？**
> 既然目標是降低操作難度，依賴套件的選擇就很明確：NumPy、SimpleITK 處理 3D 物件，pydicom 處理 DICOM 格式——都是領域內最主流、最多人用的工具。


---

📂 **查看完整差異：** [GitHub](https://github.com/higumalu/pydicomRT/commit/c8f78e6)