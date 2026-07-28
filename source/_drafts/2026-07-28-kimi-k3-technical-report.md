---
title: Kimi K3 技術報告：架構、訓練與效能深度解析
date: 2026-07-28 15:00:00
updated: 2026-07-28 15:00:00
categories: [AI, 模型分析]
tags: [Kimi, MoE, LLM, KDA, 技術報告]
description: 深度解析 Kimi K3 技術報告的核心架構創新：KDA 下限有界衰減、Stable LatentMoE、Attention Residuals，以及 2.5× 擴展效率提升的背後原理。
---

> **前言：** Kimi K3 是 Moonshot AI 開源的多兆參數 MoE 模型，2.8T 總參數、1040 億激活參數、100 萬 Token 上下文。本文基於官方技術報告，圖文並茂地解析其核心技術突破。

<!-- more -->

## 1. 概述

Kimi K3 是一個原生多模態專家混合模型（MoE），在兩個維度上同時推進前沿：

| 維度 | 內涵 |
|------|------|
| **訓練時計算擴展** | 2.8 兆總參數、1040 億激活參數、93 層、100 萬 Token 上下文 |
| **推理時計算擴展** | 多推理努力層級的強化學習、長程 Agentic 軌跡 |

相較 Kimi K2，整體擴展效率提升 **2.5×**。

---

## 2. 整體架構：三維信息流

Kimi K3 的核心設計圍繞三個互補維度組織信息流：

每個 Block 包含：
- **3 層 Kimi Delta Attention（KDA）** → 序列維度混合
- **1 層 Gated MLA** → 全局內容交互
- 每層注意力後接 **Stable LatentMoE** → 通道維度稀疏混合
- **Attention Residuals（AttnRes）** → 深度維度的跳躍連接

![Kimi K3 整體架構 Figure 2](/higumalu-note/images/k3_tech_report/page03.png)
*Kimi K3 整體架構，按 Token/Channel/Layer 三個混合維度組織，頂部為原生視覺輸入通道*

---

## 3. Kimi Delta Attention（KDA）

### 3.1 核心思想

KDA 將 **Delta Rule 遞歸**與**通道級遺忘門**結合，用固定大小的遞歸狀態 $S_t \in \mathbb{R}^{d_k \times d_v}$ 替代標準 KV Cache。

**遞歸更新（單頭形式）：**

$$S_t = (I - \beta_t k_t k_t^\top) \odot \text{Diag}(\alpha_t) \cdot S_{t-1} + \beta_t k_t v_t^\top$$

$$\tilde{o}_t = S_t^\top q_t$$

其中：
- $\alpha_t \in (0, 1)^{d_k}$：通道級 retention factor
- $\beta_t \in (0, 1)$：Delta Rule 寫入強度

### 3.2 下限有界衰減（核心創新）

**問題診斷：** Kimi Linear 的 `-Softplus` 映射無下界，16-token tile 的累積 log-decay 可能超出 `e^80`，在 BF16 中導致數值溢出。

**Kimi K3 的解法：**

$$g_t^h = g_{\min} \cdot \text{Sigmoid}(e^{A_h} z_t^h) \in (g_{\min}, 0)^{d_k}$$

$$\alpha_t^h = \exp(g_t^h) \in (e^{g_{\min}}, 1)^{d_k}$$

固定 $g_{\min} = -5$，確保每個 retention factor $\alpha > e^{-5} \approx 6.7 \times 10^{-3}$。

![下限有界衰減 Figure 3](/higumalu-note/images/k3_tech_report/page05.png)
*Log-decay 參數化對比 — Kimi Linear 無界 vs Kimi K3 有界（ gmin = −5）；對角線 tile 計算從昂貴的位置對計算變為 Tensor Core 密集運算*

### 3.3 分塊並行形式

KDA 在 chunk 級別遞歸、chunk 內並行。對於 chunk size C：

$$O[t] = \underbrace{(\Gamma_{1 \to C}^{[t]} \odot Q[t]) S[t]}_{\text {inter-chunk}} + \underbrace{A[t] \cdot eV[t]}_{\text {intra-chunk}}$$

有界衰減保證了所有因果 tile（對角線和 off-diagonal）都能用 Tensor Core 密集矩陣乘法，**完全消除了顯式位置對計算路徑**。

### 3.4 全秩輸出門

$$y_t = W_o[\text{Sigmoid}(W_g x_t) \odot \text{RMSNorm}(\tilde{o}_t)]$$

從低秩參數化升級為輸入依賴的全秩投影，增強了每個 Token 對通道讀取的調製能力。

---

## 4. Gated MLA

MLA（Multi-head Latent Attention）將每個 Token 的 Key-Value 表示壓縮到低維潛向量 $c_t = W_c x_t$，極大減少 KV Cache 佔用。

Kimi K3 的 MLA 創新：
- **No Position Encoding（NoPE）**：所有 MLA 層不應用顯式位置編碼
- **輸入依賴全秩輸出門**：$y_t = W_o[\text{Sigmoid}(W_g x_t) \odot \tilde{o}_t]$
- **FP32 注意力輸出**：修正 Flash Attention 中的有偏舍入誤差

NoPE 的意義：擴展上下文長度時無需重新調整 RoPE 頻率基或 YaRN。

---

## 5. Attention Residuals（AttnRes）

AttnRes 允許每個模塊**選擇性**從以下來源讀取表示：
- Embedding 層
- 當前 block 輸出
- 所有先前 block 輸出

通過學習的偽查詢 $w$ 生成注意力權重 $\alpha$，實現跨深度的跳躍連接，突破了傳統殘差累積的信息 bottleneck。

![擴展法則對比](/higumalu-note/images/k3_tech_report/page11.png)
*K2 與 K3 的擴展法則擬合曲線，Kimi K3 較 K2 提升 2.5× 效率*

---

## 6. Stable LatentMoE

### 6.1 規模

| 參數 | 數值 |
|------|------|
| 總專家數 | 896 |
| 每 Token 激活 | 16 |
| 共享專家 | 3 |

### 6.2 Normalized Router + Sigmoid 門控

$$\text{gate} = \text{Sigmoid}(W_g \cdot x) \in (0, 1)$$

$$s = \text{Softmax}(\text{Norm}(W_r \cdot x))$$

Normalized router 提供穩定的 Top-k 選擇，與 Sigmoid 門控配合。

### 6.3 SiTU-GLU

傳統 SwiGLU 的問題：$x > 0$ 時激活值無界。

**SiTU（平滑 Tanh 單位）：**

$$\text{SiTU-GLU}(x) = \underbrace{\beta_1 \tanh\left(\frac{W_g x}{\beta_1}\right)}_{\text{gate branch}} \odot \underbrace{\beta_2 \tanh\left(\frac{W_u x}{\beta_2}\right)}_{\text{up branch}}$$

- **Gate branch**: $\beta_1 = 4$
- **Up branch**: $\beta_2 = 25$
- **輸出嚴格有界**: $\| \text{SiTU-GLU}(x) \|_\infty \leq \beta_1 \beta_2 = 100$
- 原點附近與 SwiGLU 一階匹配（$z + O(z^3 / \beta^2)$），$\beta \to \infty$ 退化为 SwiGLU

### 6.4 Quantile Balancing（分位數平衡）

![Quantile Balancing](/higumalu-note/images/k3_tech_report/page08.png)
*(a) 不均衡路由：初始加載 (4, 3, 1, 0)；(b) 分位數平衡：每列繪製偏差後的分數灰條，紅色虛線為偏差調整線；(c) 均衡路由：最終加載 (2, 2, 2, 2)*

Quantile Balancing 從最優平衡分配問題出發，通過對偶理論證明交替最小化 token 側和專家側的分位數即可收斂到最優解。

**Histogram 估計：** 用 1000 個 bin 的直方圖估計分位數，僅需一次整數 all-reduce，代價不到原始 margin 通信的 1%。

---

## 7. 架構對比：K2 → K3

| 參數 | Kimi K2 | Kimi K3 | 變化 |
|------|---------|---------|------|
| 層數 | 61 | 93 | ↑52% |
| 總參數 | 1.04T | 2.78T | ↑167% |
| 激活參數 | 32.6B | 104.2B | ↑220% |
| Latent MoE 維度 | — | 3584 | 新增 |
| 每 Token 激活專家 | 8 | 16 | ↑100% |
| 上下文長度 | 128K | 1M | ↑8× |
| 注意力機制 | MLA | Hybrid KDA–MLA | 新架構 |
| 激活函數 | SwiGLU | SiTU-GLU | 新函數 |
| 視覺編碼器 | — | MoonViT-V2 (27層, patch=14) | 新增 |

---

## 8. 後訓練：百萬 Token 推理時擴展

### 8.1 多領域強化學習

訓練環境覆蓋：
- **長程 Coding**：SWE-Bench, Terminal-Bench, 內核優化
- **通用 Agent**：網頁開發、自主執行、工具調用（數百至數千次）
- **專業知識**：可驗證搜索
- **多模態推理**：視覺 in-the-loop 工具使用
- **持久化助理**：百萬 Token 軌跡

### 8.2 多推理努力層級

| 等級 | 含義 |
|------|------|
| Low | 快速響應 |
| Medium | 標準推理 |
| High | 深度思考 |
| Max | 最強推理能力 |

訓練時跨四個層級，蒸餾整合為統一模型。

---

## 9. 基準測試結果

![Benchmark 結果 Table 2](/higumalu-note/images/k3_tech_report/page27.png)
*跨推理、coding、agentic、vision 四大領域的基準測試對比*

### 亮點結果

| 領域 | Benchmark | Kimi K3 成績 | 排名 |
|------|-----------|-------------|------|
| **知識/推理** | GPQA Diamond | 93.5 | 🥇 全場最高 |
| **知識/推理** | CritPt | 74.7 | 🥇 全場最高 |
| **Coding** | ProgramBench | 77.8 | 🥇 全場最高 |
| **Coding** | DeepSWE | 67.5 | 第二 |
| **Agentic** | DeepSearchQA (F1) | 95.0 | 🥇 全場最高 |
| **Agentic** | BrowseComp | 93.5 | 🥇 全場最高 |
| **Vision** | OmniDocBench | 91.1 | 🥇 全場最高 |

---

## 10. 基礎設施創新

### 10.1 KDA 系統共同設計

- **FlashKDA 內核**：基於 CUTLASS，重疊 intra-chunk 計算與 cross-chunk 狀態傳播
- **KDA Context Parallelism**：跨設備分割，同時保持狀態遞歸完整性
- **狀態感知前綴緩存**：跨請求複用

### 10.2 MoonEP：完美均衡專家並行

**核心問題：** 896 專家分散在多 EP rank 時，遠程 Token 的冗餘專家拷貝導致內存爆炸。

**MoonEP 解法：** 完美均衡的專家執行，靜態計算形狀，零拷貝通信。

**理論上界：** $M(I) \leq E / R$（每 rank 冗餘專家數不超過專家數/EP ranks 數）

### 10.3 百萬 Token Agentic RL 系統

- **部分 Rollout**：長軌跡分段執行
- **外部 KV-Cache 保留**：跨分段持久化
- **可恢復 MicroVM Sandbox**：保持長壽模型和環境狀態
- **自適應節流**：動態調整執行節奏

---

## 11. 案例研究

![GPU 內核優化案例](/higumalu-note/images/k3_tech_report/page33.png)
*AttnRes 加速對比：Kimi K3 +59.7% 與 Claude Fable 5 +57.1% 並列第一*

### 11.1 GPU 內核優化

| Kernel | 提升幅度 |
|--------|---------|
| AttnRes | **59.7%** |
| DSA | 55.1% |
| KDA | 73.6% |
| MLA | >50% 峰值 TFLOPS |

Kimi K3 與 Claude Fable 5 並列第一，大幅超越 GPT-5.6 Sol（17.3%）和 GPT-5.5（30.8%）。

### 11.2 MiniTriton 編譯器開發

自主開發完整 Triton-like 編譯器：
- 自定義 tile 級 Python 前端
- Warp 級 MLIR 標注/優化層
- PTX 代碼生成管線
- 雙模式張量庫（eager + 編譯）
- 端到端訓練 GPT，loss 曲線與 PyTorch 基準偏差 < $10^{-4}$

### 11.3 晶片設計

早期概念驗證：為 nano 模型設計推理晶片原型
- 工藝：Nangate45 標準單元庫，開源 EDA 工具
- 面積：< 4 mm² @ 100 MHz
- 吞吐量：RTL 模擬 > 8,700 tokens/s

---

## 12. 總結

Kimi K3 在開源模型中達到了最高水准，其核心技術突破：

| 創新 | 意義 |
|------|------|
| **KDA 下限有界衰減** | 從昂貴位置對計算 → Tensor Core 密集運算，硬體效率大幅提升 |
| **SiTU-GLU** | 輸出嚴格有界，訓練穩定性提升，同時保留 Swish 特徵 |
| **Quantile Balancing** | 無輔助損失的數學最優均衡，histogram 估計將通信代價降至 <1% |
| **AttnRes** | 跨深度跳躍連接，突破殘差累積 bottleneck |
| **百萬 Token RL** | 首個支持長程 Agentic 軌跡的 RL 系統 |

**模型權重完全開源：** [huggingface.co/moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)

---

*本報告基於 Kimi K3 Technical Report (2026-07) 整理，圖片均來自原報告。*