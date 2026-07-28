---
title: Kimi K3 技術報告（全量翻譯 + 譯註）
date: 2026-07-28 16:00:00
updated: 2026-07-28 16:00:00
categories: [AI, 模型分析]
tags: [Kimi, MoE, LLM, KDA, 技術報告, 翻譯]
description: Kimi K3 官方技術報告全文翻譯，圖文並茂，配有譯註說明核心概念。
---

> **前言：** 本文的翻譯盡量貼近原文表述，並在必要處加入譯註說明背景概念。

<!-- more -->

## Kimi K3：開放前沿智能
### Kimi K3 技術報告
*Kimi 團隊*

---

## 摘要（Abstract）

我們提出 Kimi K3，一個 2.8 兆參數的專家混合模型，擁有 1040 億激活參數、原生視覺能力，以及 100 萬 Token 的上下文窗口。Kimi K3 基於 Kimi Delta Attention [64] 和 Attention Residuals [58] 構建，這兩項技術分別提升了序列長度和模型深度的信息流動。配合 Stable LatentMoE（每個 Token 有效激活 896 個路由專家中的 16 個）以及經過精煉的訓練配方與數據配方，這些進步帶來了相比 Kimi K2 [59] 約 **2.5 倍**的整體擴展效率提升。後訓練的亮點包括：在通用領域、Agent 領域和 Coding 領域開展強化學習，並覆蓋多個推理努力層級，實現了組合泛化和強健的長程執行能力。在 2.8 兆參數的規模下，Kimi K3 得到多個基礎設施領域的支撐：KDA 的演算法-系統共同設計、完美均衡的專家並行訓練（高效內存管理）、百萬 Token Agentic 強化學習（持久化 Rollout 與 Sandbox 狀態），以及部署創新。

廣泛的評估表明，Kimi K3 在長程 Coding、Agentic、知識、推理和視覺任務上達到了前沿水準。儘管其整體表現仍落後於最強的私有模型——Claude Fable 5 和 GPT-5.6 Sol——但 Kimi K3 一致性地超越了評估套件中的其他開源和私有模型。我們發布了完整的 Kimi K3 模型權重，以促進未來研究並加速前沿智能的更廣泛部署與應用。¹

[^1]: https://huggingface.co/moonshotai/Kimi-K3

---

## 1. 引言（Introduction）

在大型語言模型（LLM）發展的相當長一段時間裡，「scaling」意味著在部署前投入更多計算——訓練更大的模型、使用更多的數據 [55, 46]。推理模型的興起將「測試時計算」（test-time computation）確立為第二條 scaling 軸線：OpenAI 的 o 系列通過強化學習和測試時推理實現 scaling [85, 84]；Anthropic 的 Extended-Thinking 模型分配自適應思考預算並交織推理與工具使用 [6, 7]；DeepSeek-R1 [41] 和 Kimi K1.5 [119] 表明，大規模強化學習可以從強大的預訓練模型中引發複雜的推理行為；Kimi K2.5 Agent Swarm [60] 進一步將測試時 scaling 從順序推理擴展到並行 Agent 協調。這些進步使測試時 scaling 成為前沿研究的核心焦點。然而，開源模型生態系統在第二條軸線上快速進展的同時，在第一條軸線上進展緩慢：許多近期模型仍停留在 1 兆參數級別或略高 [146, 29, 136, 121]。隨著越來越複雜的推理和 Agentic 強化學習方法被應用到相似規模的預訓練基礎上，開源進展有趨同的風險，而與最強私有系統的差距正在擴大。透過 Kimi K3，我們同時追求兩條 scaling 軸線到前沿：將預訓練基礎擴展到前所未有的 3 兆參數級別，同時在 100 萬上下文長度上擴展強化學習、推理努力和長程交互。

我們提出 Kimi K3，一個原生多模態專家混合模型，總參數 2.8 兆，激活參數 1040 億，上下文窗口最長 100 萬 Token。其架構在三個互補維度上擴展信息流：
- **序列維度**：Kimi Delta Attention（KDA）[64] 提供高效的長序列混合，並以週期性插入的 Gated MLA 層保留全局交互
- **深度維度**：Attention Residuals（AttnRes）[58] 允許每層選擇性地從所有先前層的表示中檢索
- **寬度維度**：每層注意力後接 Stable LatentMoE，稀疏地激活每個 Token 的 896 個專家中的 16 個，配合 Normalization、SiTU-GLU 和 Quantile Balancing 在極度稀疏下穩定優化

這些架構進步結合精煉的數據和訓練配方，帶來相比 Kimi K2 [59] 約 2.5 倍的整體擴展效率提升。

我們將這一預訓練基礎與明確為 100 萬上下文測試時 scaling 設計的後訓練相結合。Kimi K3 在長程 Coding、通用 Agent、通用推理與知識任務上接受強化學習，每個任務涵蓋多個推理努力層級。訓練環境包括：可驗證搜索與專業知識工作、軟體工程與內核優化、視覺 in-the-loop 工具使用的多模態推理、持久化助理工作流、網頁開發，以及自主執行任務。這些環境訓練一個通用的推理-行動-觀察-驗證-適應循環，往往跨越數百或數千次工具調用和數百萬累計上下文 Token。領域和努力級別專門的策略透過多教師 On-Policy 蒸餾 [76, 135, 29] 整合到統一模型中。

實現這一 regime 需要能隨架構複雜度、模型規模和軌跡長度擴展的基礎設施。針對 KDA 的系統共同設計，我們開發了融合內核、KDA Context Parallelism 和狀態感知前綴緩存，使 KDA 在設備內、跨設備和跨請求上都高效。對於 2.8 兆參數 MoE 預訓練，MoonEP 提供完美均衡的專家執行（靜態計算形狀、零拷貝通信），配合內存高效訓練和多模態編碼器優化，在有限內存下維持高利用率。針對百萬 Token Agentic 強化學習，我們的協作系統結合部分 Rollout、外部 KV-Cache 保留、自適應節流和可恢復 MicroVM Sandbox，以保持長壽模型和環境狀態。最後，專門化內核與緩存/預算感知調度將這些創新轉化為可預測的生產服務。

由此產生的模型建立了新的開放前沿。在涵蓋長程 Coding、Agentic、知識、推理和視覺任務的基準測試上，Kimi K3 總體落後於最強的私有系統——Claude Fable 5 和 GPT-5.6 Sol——但一致性地超越了評估套件中的其他開源和私有模型（如圖 1 所示）。

我們的貢獻總結如下：

- **開放前沿的預訓練。** 我們訓練了一個 2.8 兆參數的原生多模態 MoE 模型，激活參數 1040 億，上下文窗口 100 萬 Token。KDA、AttnRes、Stable LatentMoE 和精煉的數據/訓練配方共同帶來相比 Kimi K2 約 2.5 倍的整體擴展效率提升。
- **多努力層級推理時 scaling 的強化學習。** 我們在通用、Agentic 和 Coding 領域以及多個推理努力層級上開展強化學習，然後將結果能力整合到統一模型中。
- **多兆參數、百萬 Token 智能的基礎設施。** 我們提出 KDA 系統共同設計；MoonEP 和 2.8 兆參數 MoE 預訓練的內存高效基礎設施；帶可恢復沙箱的協作強化學習系統（處理百萬 Token Agentic 軌跡）；以及更多基礎設施創新。
- **一個開放前沿模型。** 我們發布完整的 Kimi K3 模型權重，使前沿智能可用於研究、部署和進一步創新。

---

## 2. 模型架構（Model Architecture）

Kimi K3 的架構設計為在三個互補維度上擴展信息流：**序列長度、網絡深度、模型寬度**。

- **序列維度**：Hybrid Attention 將三層 Kimi Delta Attention（KDA）[64] 與一層 Gated MLA 結合，在提供高效長上下文 Token 混合的同時保留選擇性高容量注意力（§2.1）。
- **深度維度**：Attention Residuals（AttnRes）[58] 允許每個模塊選擇性地從 Embedding、當前 block 和先前 block 中檢索表示，將信息訪問擴展到傳統順序殘差累積之外（§2.2）。
- **寬度維度**：每層注意力後接 Stable LatentMoE 層，稀疏地執行通道混合，每個 Token 有效激活 896 個路由專家中的 16 個（§2.3）。
- **原生視覺**：MoonViT-V2 編碼圖像和視頻，輕量級投影器將視覺特徵映射到共享 Embedding 空間，再送入骨幹網絡處理（§2.4）。
- 配合 Per-Head Muon 優化器（§2.5），這些組件提供了跨 Token、層和通道擴展信息流的統一架構。

結合精煉的數據和訓練配方，它們帶來相比 Kimi K2 約 **2.5 倍**的整體擴展效率提升。圖 2 提供了架構總覽。

![Kimi K3 整體架構](/higumalu-note/images/k3_tech_report/architecture.png)
*Kimi K3 整體架構，按 Token、Channel 和 Layer 三個混合維度組織，輸入端有原生視覺通道。*

---

### 2.1 Hybrid Attention

Kimi K3 使用層級混合的線性注意力和全局注意力，結合 KDA [64] 與 Gated MLA。每個 Block 包含 3 層 KDA 後接 1 層 Gated MLA，給出 3:1 的混合比例。此模式在整個骨幹網絡中重複。兩種注意力機制將在下面分別描述。骨架末端追加一個額外的 Gated MLA 層，確保最終層始終執行全局注意力。

#### 2.1.1 Kimi Delta Attention

KDA 在 Delta Rule 遞歸 [106, 139] 的基礎上增加通道級遺忘門 [64]。考慮隱藏狀態序列 $x_t \in \mathbb{R}^d$，其中 $t$ 索引 Token 位置，$d$ 是模型隱維度。為清晰起見，我們先描述單個注意力頭，Query 和 Key 向量 $q_t, k_t \in \mathbb{R}^{d_k}$，Value 向量 $v_t \in \mathbb{R}^{d_v}$，以及遞歸狀態 $S_t \in \mathbb{R}^{d_k \times d_v}$。KDA 在 Delta Rule 更新前應用通道級衰減：

$$S_t = (I - \beta_t k_t k_t^\top) \odot \text{Diag}(\alpha_t) \cdot S_{t-1} + \beta_t k_t v_t^\top, \quad \tilde{o}_t = S_t^\top q_t$$

其中 $\alpha_t \in (0, 1)^{d_k}$ 是通道級單步 retention factor，$\beta_t \in (0, 1)$ 控制 Delta Rule 寫入強度。

跟隨 Kimi Linear [64]，KDA 將每頭量參數化為：

$$q_t^h, k_t^h = \text{L2Norm}\left(\text{Swish}\left(\text{ShortConv}\left(W_t^{q/k} x_t\right)\right)\right) \in \mathbb{R}^{d_k}$$

$$v_t^h = \text{Swish}\left(\text{ShortConv}\left(W_t^v x_t\right)\right) \in \mathbb{R}^{d_v}$$

$$\beta_t^h = \text{Sigmoid}\left(W_\beta x_t\right) \in (0, 1)$$

$$z_t^h = W_\alpha^\uparrow W_\alpha^\downarrow x_t + b_\alpha^h \in \mathbb{R}^{d_k}$$

Query、Key、Value 投影依次應用 ShortConv 和 Swish [139]，Query 和 Key 進一步以 L2Norm [142] 歸一化。低秩投影和頭特定偏置 $b_\alpha^h \in \mathbb{R}^{d_k}$ 為每個 Key 通道產生細粒度衰減 logit $z_t^h$。從 $z_t^h$ 到 $\alpha_t^h$ 的下限有界映射將在分塊並行形式之後引入。

**分塊並行形式**

跟隨 Kimi Linear [64]，KDA 在 chunk 級別遞歸，在每個 chunk 內並行。對於 chunk size $C$，$X[t]$ 堆疊第 $t$ 個 chunk 中的 Token 向量（$X \in \{Q, K, V, O, U, W\}$）。矩陣 $S[t] \in \mathbb{R}^{d_k \times d_v}$ 表示進入 chunk $t$ 的遞歸狀態。對位置 $1 \leq i \leq j \leq C$，定義通道級累積衰減：

$$\gamma_{i \to j}^{[t]} := \prod_{r=i}^j \alpha_r^{[t]}, \quad \gamma_r^{[t]} := \gamma_{1 \to r}^{[t]}$$

如 Kimi Linear 中，$\Gamma_{1 \to C}^{[t]} \in \mathbb{R}^{C \times d_k}$ 將 $\gamma_1^{[t]}, ..., \gamma_C^{[t]}$ 按行堆疊。UT 變換產生 $U[t]$ 和 $W[t]$，從而定義偽 Value 項 $eV[t] := U[t] - W[t]S[t]$。給定傳入狀態 $S[t]$，chunk $t$ 中的所有輸出並行計算為：

$$A[t] = \text{Tril}\left((Q[t] \odot \Gamma_{1 \to C}^{[t]}) \cdot (K[t] / \Gamma_{1 \to C}^{[t]})^\top\right)$$

$$O[t] = \underbrace{(\Gamma_{1 \to C}^{[t]} \odot Q[t]) S[t]}_{\text{inter-chunk}} + \underbrace{A[t] \cdot eV[t]}_{\text{intra-chunk}}$$

對於矩陣 $M$，$\text{Tril}(M)$ 將所有嚴格上三角 entries 置零，保留下三角 entries（包括對角線）。此 mask 在 chunk 內強制執行因果交互，對角線保留是因為每個輸出讀取當前 Token 更新後的狀態。$O[t]$ 的第一項承載來自先前 chunks 的信息，第二項處理當前 chunk 內的交互。

**下限有界衰減（核心創新）**

方程 4 在每個 chunk 中以倒數累積衰減 $1/\Gamma_{1 \to C}^{[t]}$ 重新縮放 Keys。由於 $\Gamma_{1 \to C}^{[t]}$ 是 $(0, 1)$ 內 retention factor 的乘積，其倒數可能無界、在有限精度下溢出 [141, 64]。Kimi Linear 透過在 log 空間計算相對衰減並將每個 chunk 進一步劃分為次級 16-token tile 來控制數值範圍 [141, 64]。Off-diagonal tile 可以直接用 Tensor Core 密集矩陣乘法計算；相比之下，diagonal tile 仍需要顯式位置對計算，這是主要的 intra-chunk 瓶頸。

Kimi K3 透過改變從衰減 logit $z_t^h$ 到單步 log-decay $g_t^h$ 的映射來解決這一瓶頸。跟隨 GDN 和 Mamba-2，Kimi Linear 使用負 Softplus 映射 $g_t^h = -e^{A_h} \text{Softplus}(z_t^h) \in (-\infty, 0)^{d_k}$ [139, 24, 64]。Kimi K3 轉而使用縮放 Sigmoid 從下方約束 log-decay：

$$g_t^h = g_{\min} \cdot \text{Sigmoid}(e^{A_h} z_t^h) \in (g_{\min}, 0)^{d_k}$$

$$\alpha_t^h = \exp(g_t^h) \in (e^{g_{\min}}, 1)^{d_k}$$

其中 $A_h$ 是每頭可學習 log-scale，$g_{\min} = -5$ 固定。我們初始化 $A_h = 0$，各偏置 $b_\alpha^h$ 按 [64, 24, 139] 初始化。$g_{\min} = -5$ 確保每個 retention factor $\alpha_{t,j}^h > e^{-5} \approx 6.7 \times 10^{-3}$，16-token tile 上的累積 log-decay 位於 $(-80, 0)$。相應的倒數重新縮放因子因此小於 $e^{80}$，保持在 BF16 動態範圍內。這一有限範圍使對角線和 off-diagonal tile 都能使用密集 Tensor Core 矩陣乘法，消除了位置對角線路徑。

![KDA 下限有界衰減 Figure 3](/higumalu-note/images/k3_tech_report/kda-decay.png)
*圖 3：(a) Log-decay 參數化：Kimi Linear 使用無界負 Softplus 映射，Kimi K3 以縮放 Sigmoid 約束 log-decay 下界；(b) 對角線 tile 計算：Kimi Linear 對每個對角線 tile 執行顯式位置對計算，而 Kimi K3 的有界範圍使所有因果 tile 使用密集 Tensor Core 矩陣乘法*

**全秩輸出門**

最後，Kimi K3 將 KDA 的輸出門從 Kimi Linear [64] 使用的低秩參數化改為輸入依賴的全秩投影。對遞歸輸出應用頭級 RMSNorm [147] 後，KDA 應用數據依賴輸出門控 [100]：

$$y_t = W_o[\text{Sigmoid}(W_g x_t) \odot \text{RMSNorm}(\tilde{o}_t)]$$

#### 2.1.2 Gated MLA

Multi-head Latent Attention（MLA）在 DeepSeek-V2 [28] 中引入，將每個 Token 的 Key-Value 表示壓縮到低維潛向量 $c_t = W_c x_t$。MLA 不緩存完整的頭特定 Keys 和 Values，而是緩存 $c_t$ 並在注意力計算期間透過學習的上投影重建內容 Keys 和 Values。這種分解減少了 KV-Cache 佔用，同時保留了全局 Token 間注意力。MLA 後被 Kimi K2 和 Kimi K2.5 [59, 60] 採用，Kimi K3 在週期性全局注意力層中保留 MLA。

不同於 Kimi K2 和 K2.5，Kimi K3 跟隨 Kimi Linear [64] 的混合設計，對所有 MLA 層應用 **No Position Encoding（NoPE）**。因此，不對它們的 Query 或 Key 應用顯式位置編碼。穿插的 KDA 層提供位置敏感和近因感知的序列混合，而 MLA 層提供無限制的全局內容交互。這種分離也避免了在擴展上下文長度時修改位置編碼參數，例如重新調優 RoPE 頻率基或應用 YaRN [93]。

此外，Kimi K3 以輸入依賴、通道級全秩輸出門增強 MLA。令 $\tilde{o}_t$ 為位置 $t$ 的無門控 MLA 輸出，門控輸出為：

$$y_t = W_o[\text{Sigmoid}(W_g x_t) \odot \tilde{o}_t]$$

門投影 $W_g$ 是全秩的，與 Kimi K3 中 KDA 的新參數化一致。此門允許每個 Token 調製從全局注意力讀取的通道。

為修正 Flash Attention 中出現的有偏舍入誤差，我們採用 [99] 的方法，在訓練期間將注意力輸出保持為 FP32。這使輸出 tile 的片上佔用增加一倍；因此我們重新設計訓練內核，讓它與 KV 緩衝區而不是 Query tile 重疊，為更深的 KV 管線和更高訓練吞吐量釋放共享內存。

---

### 2.2 Attention Residuals

標準殘差連接 [44] 將所有先前信息壓縮到深度上的單一狀態 $h_l$——這是一種令人聯想到時間維度 RNN 的 bottleneck。對於序列建模，Transformer 以注意力 [10, 126] 取代了遞歸，允許每個位置以數據依賴權重選擇性地訪問所有先前位置。Attention Residuals（AttnRes）[58] 將相同方法論應用於深度維度：每層選擇性地從所有先前層的表示中檢索，而非均勻累積。

**完全 Attention Residuals**

對於每層 $l$，我們定義層特定可學習偽查詢 $q_l = w_l \in \mathbb{R}^d$ 及其 Keys 和 Values：

$$k_i = v_i = \begin{cases} h_1 & i = 0 \\ f_i(h_i) & 1 \leq i \leq l-1 \end{cases}$$

其中 $f_i(h_i)$ 是第 $i$ 層的輸出，$h_1$ 是 Token Embedding。注意力權重遵循 softmax kernel $\phi(q, k) = \exp(q^\top \text{RMSNorm}(k))$ [56, 147]，其中 RMSNorm 防止大幅度輸出的層主導權重：

$$\alpha_{i \to l} = \frac{\phi(q_l, k_i)}{\sum_{j=0}^{l-1} \phi(q_l, k_j)}, \quad h_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot v_i$$

由於網絡深度適中（$L < 100$），這種完全形式 $O(L^2 d)$ 的算術代價是可接受的；實際開銷是為保持所有層輸出存活所需的 $O(Ld)$ 內存（以及管道並行下跨階段通信）。

**分塊 Attention Residuals**

為減少此開銷，我們將 $L$ 層劃分為 $N$ 個 block，每個 block $S = L/N$ 層。Block $n$（層索引 $B_n$）內的層輸出通過求和歸約為單一表示 $b_n = \sum_{j \in B_n} f_j(h_j)$，其中 $b_n^i$ 表示 block 前 $i$ 層的部分和；設 $b_0 = h_1$，因此 Token Embedding 始終作為來源包含在內。跨 block，只對 $N$ 個 block 級表示應用完全注意力：對 block $n$ 的第 $i$ 層，Value 矩陣為：

$$V = \begin{cases} [b_0, b_1, ..., b_{n-1}]^\top & i = 1 \text{（block 首層）} \\ [b_0, b_1, ..., b_{n-1}, b_{n-1}^i]^\top & i \geq 2 \text{（後續層）} \end{cases}$$

---

### 2.3 Stable LatentMoE

LatentMoE 在前饋網絡中引入潛路由，將 FFN 轉換為潛專家混合。路由的 Key-Value 表示被壓縮為低維向量，激活時通過上投影重建。具體來說，令 $x \in \mathbb{R}^d$ 為輸入，路由器首先計算：

$$s = \text{Softmax}(\text{Norm}(W_g x)) \in \mathbb{R}^n$$

其中 $n = 896$ 是路由專家數，Norm 是可學習的通道級 Normalization。為提高路由穩定性，門控分支計算：

$$\text{gate} = \text{Sigmoid}(W_g x) \in (0, 1)$$

最終權重為 $w = \text{gate} \odot s$。為處理路由中的不平衡，我們引入每專家偏置 $b \in \mathbb{R}^n$，並定義 Top-k 選擇和路由權重（依 Eq. 10）：

$$s_i = \text{Sigmoid}(W_g x_i), \quad \mathcal{T}_i = \text{argtop}_k(s_i + b), \quad p_{i,j} = \begin{cases} \dfrac{s_{i,j}}{\sum_{r \in \mathcal{T}_i} s_{i,r}} & j \in \mathcal{T}_i \\ 0 & \text{otherwise} \end{cases}$$

偏置 $b$ 從路由概率 $p_{i,j}$ 計算中省略，這樣它只調節 dispatch（Top-k 選了哪些專家）而不改變混合權重或直接干擾路由器的梯度優化。

#### 2.3.1 SiTU-GLU

在 FFN 投影之間，我們用 SiTU-GLU 取代標准 SwiGLU。SiTU-GLU 將 Swish 激活函數的 cap 替換為平滑的 Tanh cap：

$$\text{SiTU-GLU}(x) = \beta_1 \tanh\left(\frac{W_g x}{\beta_1}\right) \odot \beta_2 \tanh\left(\frac{W_u x}{\beta_2}\right)$$

Gate branch 的 $\beta_1 = 4$，Up branch 的 $\beta_2 = 25$。由於 $|\tanh(z)| < 1$ 且 $0 < \text{Sigmoid}(z) < 1$，每個輸出坐標滿足 $\|\text{SiTU-GLU}(x)\|_\infty \leq \beta_1 \beta_2 = 100$，輸出有嚴格上界。在原點附近，$\beta \tanh(z/\beta) = z + O(z^3 / \beta^2)$，因此 SiTU-GLU 與 SwiGLU 一階匹配。隨 $\beta \to \infty$ 精確退化為 SwiGLU。

#### 2.3.2 Quantile Balancing

隨著專家數增長到近千，均衡路由變得越來越重要。傳統輔助損失方法 [30] 在每個 Token 上增加一個負載不平衡罰項，但這會直接干擾路由決策與梯度優化。我們採用無輔助損失的路由均衡：對每個專家維護一個偏置項 $b_j$，根據目標加載 $q := mk/n$ 週期性更新（依 Eq. 14）：

$$b_j \leftarrow b_j - \eta \left( \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}_{j \in \mathcal{T}_i} - \frac{k}{n} \right)$$

此更新將專家加載推向目標，類似於 SignSGD，但直接作用於偏置而非路由器參數。

![Quantile Balancing](/higumalu-note/images/k3_tech_report/quantile-balancing.png)
*圖 5：Quantile Balancing 示意（m=8 tokens, n=4 experts, k=1）。(a) 不均衡路由：初始加載 (4, 3, 1, 0)；(b) 分位數平衡：每列繪製偏差後的分數灰條，紅色虛線為偏差調整線；(c) 均衡路由：最終加載 (2, 2, 2, 2)*

對於精確的偏差估計，我們從最優平衡分配問題的對偶推导出 Alternating Quantile Solver：交替求解 token 側閾值 $\alpha$ 和專家側閾值 $\beta$，每步封閉形式精確求解。實務上，用 1000 個 bin 的直方圖估計分位數，僅需一次整數 All-Reduce 通信，代價不到原始 margin 通信的 1%。

---

### 2.4 原生視覺

Kimi K3 的視覺編碼器 MoonViT-V2 將圖像和視頻編碼為視覺 Token 序列。輕量級投影器將視覺特徵映射到與語言骨幹共享的 Embedding 空間，然後一起送入骨幹網絡處理。這種原生多模態設計使視覺理解與語言推理深度融合，而非事後拼接。

此視覺訓練配方基於 Kimi K2.5 [60, 62] 的整體設計構建：視覺輸入首先由 MoonViT-V2 編碼，然後由輕量級 MLP 投影器映射到共享語言空間。MoonViT-V2 是一個約 4 億參數的 27 層視覺 Transformer，採用 RMSNorm 並從其線性投影和注意力投影中移除所有偏置項——此設計從訓練初期就進一步穩定了跨模態表示融合。

圖像和視頻以完全共享的參數處理：注意力分為幀內空間和幀間時間兩遍，時間池化進一步壓縮 Token 維度。投影前，像素洗牌（pixel-shuffle）操作以 2×2 下採樣將 Token 數量減少到原來的四分之一，使最長 3584×3584 像素的輸入在百萬 Token 上下文窗口內仍然可負擔。

---

### 2.5 Per-Head Muon

我們對注意力頭使用 Per-Head Muon 優化器。Muon（Matrix u for u, Newton-ish）是一種為 Transformer 設計的牛頓類優化器，在某些設置下收斂速度比 AdamW 快 [22]。Per-Head 版本對每個注意力頭維護單獨的學習率和狀態，進一步提升收斂穩定性。

---

## 3. 預訓練（Pre-Training）

### 3.1 數據配方

Kimi K3 的預訓練數據包括多語言語料、專業領域文本（代碼、數學、科學論文）和網頁內容。關鍵改進包括：
- **數據質量過濾**：多階段質量分類器去除低質量內容
- **領域平衡**：確保數學、編程、推理密集內容的充分覆蓋
- **去重**：大規模語義去重減少冗餘

### 3.2 訓練穩定性

2.8 兆參數規模的訓練面臨獨特挑戰：
- **權重裁剪**：從 Kimi K2 沿用的權重裁剪機制防止梯度爆炸
- **漸進式課程學習**：從短序列開始，逐步過渡到百萬 Token
- **BF16/FP32 混合精度**：關鍵操作使用 FP32 防止累積誤差

### 3.3 擴展法則

![擴展法則](/higumalu-note/images/k3_tech_report/scaling-law.png)
*圖 7：Kimi K2 和 Kimi K3 的擬合擴展法則曲線。Kimi K3 在相同 FLOPs 下達到更低驗證損失，相比 Kimi K2 提升 2.5× 擴展效率*

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

### 3.4 長上下文擴展

Kimi K3 的預訓練從 8K Token 上下文長度開始，後在退火階段逐步擴展到 64K Token。

**位置編碼**：Kimi K3 不使用任何顯式位置嵌入（NoPE），而是透過 KDA 的遞歸門控和衰減機制隱式編碼位置信息。因此，模型無需任何位置編碼修改（如 RoPE 重縮放或插值 [93]）即可直接外推到百萬 Token 上下文。

**長上下文數據**：來自自然來源的長文檔和視頻含有大量低質量內容，包括近似重複項、二進制 blob、截斷文件、視頻剪輯和無效的機器生成文本。我們因此透過專門的清理流程處理它們：結合精確和模糊去重（視頻幀基於感知哈希），輔以啟發式和分類器的質量過濾，以及自動化驗證。由於真正長且連貫的文檔和視頻相比短文本稀缺，我們對其進行上採樣，使長上下文分佈在退火期間不被短序列淹沒。然而，長度本身並不賦予長程能力。為解決這個問題，我們透過打散和拼接多模態文檔和子任務來合成額外的長上下文數據，使嵌入的任務只能在注意分散在完整百萬 Token 上下文中的信息時才能解決。這使注意力機制在完整尺度上接受訓練，防止其退化為局部模式。

**漸進式上下文擴展**：Kimi K3 支持最長百萬 Token 的上下文窗口。我們透過隨訓練進展逐步擴展上下文窗口來實現這一點，遵循四階段課程。窗口在預訓練期間從 8K 擴展到 64K Token，並在退火階段從 256K 擴展到 1M Token。將昂貴的長序列計算集中在整體訓練預算的小部分中，使課程保持經濟性的同時仍允許模型逐步適應越來越長程的依賴關係。讓百萬 Token 訓練對 KDA 層可行的序列維度分區在 §5.1 中描述。

---

## 4. 後訓練（Post-Training）

### 4.1 強化學習

Kimi K3 的後訓練明確為 100 萬上下文測試時 scaling 設計。跨四個領域開展強化學習，每個領域覆蓋多個推理努力層級（low / medium / high / max）。

**通用領域**：可驗證搜索、專業知識工作，訓練模型在長上下文中的精確信息檢索與綜合能力。

**Agentic 領域**：持久化助理工作流、網頁開發、自主執行任務，訓練數百至數千次工具調用、數百萬累計上下文 Token 的長程執行能力。

**Coding 領域**：軟體工程、內核優化，訓練模型在複雜代碼庫中的導航、修改和調試能力。

**多模態領域**：視覺 in-the-loop 工具使用，訓練模型結合視覺理解和工具調用的能力。

**領域專門化策略的統一**：每個領域和努力層級的專門化策略透過多教師 On-Policy 蒸餾 [76, 135, 29] 整合到統一模型中。

**演算法：Partial Rollout**

為緩解長程任務中加劇的長尾延遲，我們從同步 RL 框架 [119, 60] 擴展了 partial rollout 方案。在每次迭代的 rollout 階段，我們對每個 prompt 採樣 K 個 completion，維護 N × K 條軌跡的活跃工作負載。不等待所有 rollout 終止，一旦 λ × N × K 條軌跡完成（即 λ ∈ (0, 1) 的 fraction），生成階段就暫停，讓策略優化在無 straggler 的情況下繼續。暫停的 rollout 被排隊並在下次迭代開始時優先恢復，由 sandbox 基礎設施驅動（§ 5.3.2）。當 prompt 的 K 個 response 全部完成後，立即分發進行策略優化，遵循 Kimi K2.5 [60] 的演算法。

在 partial rollout 方案下，個別長程軌跡自然跨越多個迭代，引入威脅訓練穩定性的數據陳舊性。我們的策略優化演算法透過 per-token 正則化先天容忍此類極端 off-policy regime。透過將策略更新約束在局部鄰域內，此正則化使演算法能穩健處理高度陳舊的數據並維持訓練穩定性。

**推理努力 RL**

為在最大化 token 效率的同時微調推理努力，我們在 RL 期間 [60] 實現 per-problem 預算控制機制。每個問題 x 關聯一個初始 token 預算 b₀(x)，從冷啟動模型估計，當軌跡總 token 預算 T(y) 超過縮放閾值 τ · b₀(x) 時，以 −1 覆蓋任務獎勵。對於通用任務，T(y) 測量思考 token 數；對於 agentic 任務，T(y) 包含累計輸出 token（推理軌跡 + 工具調用參數）。訓練遵循 τ 的分階段課程：先用相對大的 τ 訓練 max-budget 變體（仍設置上限抑制過度 overthinking），然後將 τ 退火至更小值得到 high- 和 low-effort 專家模型。τ 的調整由人類在迴指導下 per-domain 配置。各推理層級專家生成的軌跡共同收集，用於監督微調和多教師 On-Policy 蒸餾。

**Agentic 生成獎勵模型**

對於不可驗證的通用任務，我們採用 Agentic Generative Reward Model（GRM），保留 Kimi K2.5 [59, 60] 中的小組獎勵 tournament 風格二值比較。超越通用 agentic 能力的增強判斷，agentic judge 需遵循強制協議：(1) 閱讀結果、產品或文本輸出；(2) 生成評分 rubrics；(3) 根據 rubrics 對每個候選評分；(4) 將 rubrics 分配的分數記錄在 scorepad 中。為緩解獎勵 hack 趨向越來越冗長輸出，我們應用類似推理努力控制的基於預算的冗長度控制：給定從冷啟動模型估計的初始冗長度 ℓ₀ 和倍增器 σ，輸出長度超過 σ · ℓ₀ 的候選自動失去二值比較。

**多教師 On-Policy 蒸餾**

我們採用 Multi-Teacher On-Policy Distillation（MOPD）在不同推理努力下整合這些領域專門化能力到統一模型 [76, 135, 29]。訓練時，對於給定領域 d 和採樣的推理努力層級 e ∈ {low, high, max}，優化由對應的教師模型 π(d,e) 引導。給定輸入查詢 x 和前綴回應 y<t，在 yt 上評估的 per-token OPD 獎勵為蒸餾信號，確保統一模型繼承各領域、各努力層級的能力。

**OPD 獎勵公式**（Eq. 15）：每個 token yt 的 OPD 獎勵定義為：

$$r_{\text{opd}}(y_t | e, x, y_{<t}) = \text{clip}\left(\frac{\text{sg}(\log \pi^{(d,e)}_{\text{teacher}}(y_t | x, y_{<t}))}{\pi_\theta(y_t | e, x, y_{<t})}, -R_{\max}, R_{\max}\right)$$

其中 sg(·) 為 stop-gradient 算子，Rmax > 0 為裁剪閾值。此密集獎勵信號無縫集成到我們的 RL 框架中，自然支持長程任務的 partial rollout 訓練等基礎設施優化。

### 4.1.4 部署感知後訓練

**MXFP4 量化感知後訓練**：為減少部署時的內存佔用和服務成本，我們將佔模型參數內存主導的 MoE 專家權重量化到 MXFP4 [104]，激活以 MXFP8 計算；而所有非專家組件（注意力投影、潛 MoE 投影、共享專家和 MoE 路由器）保持在更高精度。我們在整個後訓練階段——涵蓋 SFT 和 RL——執行量化感知訓練（QAT）[50]，使模型適應量化引起的精度損失。在 RL 期間，rollout 和訓練共享相同的量化方案，消除訓練-推理失配。

**Draft Model 微調**：優化複雜、長程 agentic 模型的推理效率至關重要。Kimi K3 預訓練時配有一個多 Token 預測（MTP）層，其結構與骨幹 block 相同。由於 EAGLE-3 [72] 的 draft model 由單一解碼器層組成，其結構與 MTP 層匹配，我們將預訓練的 MTP 層微調為 EAGLE-3 風格的 draft model，目標模型保持凍結，只更新 draft 層及其特徵融合投影。遵循 EAGLE-3 的訓練時測試協議，draft 在訓練時展開 7 步；在第一步之後（新 token 的目標側特徵不可用），draft 消耗自身先前步的輸出，模擬推理中的循環 draft 過程。

Draft 輸入融合目標模型的低層、中層和高層特徵，分別取自第 1 個、中間和最終 AttnRes block 的輸出（§2.2）。這些特徵通過無偏置矩陣 $W_{E3}$ 拼接投影到 draft 層的隱藏大小，初始化為 $\begin{bmatrix} 0 & 0 & I \end{bmatrix}$，使融合表示在初始化時與高層特徵 $h_h$ 一致——這是 MTP 層預訓練時的輸入——並在微調過程中逐步學會融合低層和中層特徵。

推測解碼的加速取決於無損推測採樣下每 token 接受率 $P = \sum_{x \in V} \min(p(x), q(x))$，其中 p 和 q 分別是目標和 draft model 的下一 token 分佈。由於傳統 KL 散度代理不能保證最大化容量有限的 draft model 的接受率，我們直接優化似然基於的 LK 損失 [105]——接受率本身的負對數：

$$L_{LK} = -\log \sum_{x \in V} \min(p(x), q(x))$$

p 和 q 在 temperature 1 下評估，無輔助真實交叉熵項。Draft 微調遵循後訓練 QAT 配置（§4.1.4），MoE 專家權重為 MXFP4，其輸入激活為 MXFP8，而非專家模組保持更高精度。

### 4.2 Long-Context SFT

在強化學習之前，Kimi K3 在長上下文任務上接受監督微調（SFT），涵蓋長程推理、多文檔問答和代碼補全。關鍵是使用真實的長上下文數據而非簡單的序列拼接。

### 4.3 知識圖譜引導的任務合成

**動機與概述**

後訓練任務的質量和多樣性很大程度上取決於其原材料。細粒度概念檢索引導的檢索能挖掘專業化和代表性不足的知識，而跨多樣概念採樣拓寬領域覆蓋範圍。為在大規模下同時控制粒度和覆蓋範圍，我們構建了一個自我演化、層次組織的知識圖譜，讓 agent 透過網路規模探索持續擴展跨知識密集型和 Coding 領域的覆蓋範圍。

**Agentic 知識圖譜構建**

我們透過遞歸、agent 驅動的擴展將知識圖譜構建為有向無環圖。擴展過程從一組預定義的粗粒度種子節點開始。每個節點分配一個 agent 實例，執行多次網路搜索以調研相應概念。在添加新節點之前，agent 會探索現有圖譜以識別等價或相關概念，在適當處復用現有節點並最小化重複。邊總是從較粗概念指向較細概念，無論 agent 先發現哪個端點。新添加的節點後續分配給 agent 進一步探索。當分配的 agent 判斷當前概念足夠原子化時，分支停止擴展。

**材料檢索與任務合成**

為瞄準跨領域和任務類型的期望分佈，系統以不同粒度級別採樣節點（單獨或相關組合）。從採樣節點衍生的關鍵詞與其祖先在知識圖譜中的上下文信息相結合，形成網路查詢。檢索到的真實材料被組裝，以便合成 agent 生成各種類型的訓練任務。

![知識圖譜引導的任務合成流程](/higumalu-note/images/k3_tech_report/knowledge-graph-synthesis.png)
*圖 9：知識圖譜引導的任務合成流程概覽。層次組織的知識圖譜代表跨多個粒度級別的概念，從廣義領域到細粒度概念。採樣相關節點形成關鍵詞集，用於從互聯網檢索公開材料。每次合成實例中，系統選擇一種任務類型，並利用材料合成相應的訓練任務。*

### 4.2.1 統一白盒 RL 環境

使用單一固定 agent harness 進行訓練會導致模型過擬合於特定工具 schema、系統提示、上下文管理機制或交互協議。為解決這個問題，我們開發了一個統一白盒 RL 環境，將 agent harness 表示為可配置、可組合模組的集合，包括工具接口、系統提示、上下文管理策略、技能、記憶、子 agent 及其他組件。透過配置動態組合這些模組，環境可以實例化主流 harness 如 Kimi Code [57]、Claude Code [15]、Codex [20]、OpenClaw [87] 和 Hermes [45]，以及全新 harness。在 RL 訓練期間，我們動態為不同任務組構造不同的 harness 配置，使 Kimi K3 接觸多樣化的模組組合而非單一 harness 的約定。這種抽象同樣可以輕鬆支持跨多個任務領域的 RL，為訓練更通用 agent 提供可擴展的基礎。

### 4.2.3 Agentic 環境中的可驗證問題

我們在 agentic 環境中的可驗證問題上訓練 Kimi K3；典型例子包括：多步複雜信息搜索，模型規劃研究、逐步從網路收集證據並產生可驗證答案；專業人士的日常工作，如投資銀行、數據分析和法律實踐，模型在其中分解複雜請求、在沙盒中操作領域工具，並在數十到數百步中完成複雜交付物；以及多步可驗證的視覺推理，涵蓋 STEM 問題、視覺謎題和圖表理解。每個視覺推理軌跡在配備隔離沙盒中 Python 解釋器的 agent 環境中生成：模型迭代編寫並執行代碼以裁剪、縮放或以其他方式變換輸入圖像、執行精確計算或驗證中間結果，並將執行輸出——包括生成的圖像——作為多個交互步驟中的新觀察。隨著模型學會執行適當的圖像操作並收集更多觀察，其在複雜視覺推理任務上的表現穩步提升。

### 4.2.4 GPU 內核優化任務

為加強 Kimi K3 的 GPU 內核優化能力，我們構建了一個大規模內核任務套件，從單算子內核到融合 mega-內核，源自我們自己和其他高質量 GitHub 倉庫（如 Flash Linear Attention [140]）。該套件涵蓋多樣的 GPU 編程方法，如 CUDA、Triton、CuTe DSL、Gluon、ThunderKittens [111] 和 TileLang [130]，覆蓋廣泛使用的 GPU 架構和數值格式，包括 FP16、BF16、FP8 和 FP4。獎勵同時評估正確性和性能：每個內核提供 PyTorch 參考實現，超過預定義數值誤差閾值的解決方案獲得零獎勵。性能相對於專家實現評分：匹配則獎勵 0.5，接近硬體極限則獎勵趨近 1。為確保獎勵反映真正的優化，我們開發了一個 hack 檢測系統，對 CUDA graph replay、輸入緩存和精度降低等獎勵 hack 策略進行懲罰，並在 Kimi K3 開發過程中持續用新safeguard 擴展。

### 4.2.5 個人助理任務

對於長程個人助理任務，我們開發了 Gmail、Notion、Slack 和 Canvas 等廣泛使用應用的現實模擬實現。它們保留真實對應物的核心語義，同時支持可重現的大規模交互而無需外部 API 或速率限制。在此模擬應用基礎上，我們設計了人力資源、法律服務和財務等場景中現實專業工作流程啟發的複雜任務。在每個任務中，agent 在持久化、演化的環境中跨多個模擬天數運營，遇到新信息並需相應調整其行為。

---

## 5. 基礎設施（Infrastructure）

Kimi K3 結合了三種罕見的系統挑戰：混合 KDA 注意力、3 兆類稀疏多模態訓練/推理，以及百萬 Token Agentic 工作負載。我們的基礎設施在每個領域都進行了共同設計。

### 5.1 KDA 演算法-系統共同設計

KDA 以固定大小遞歸狀態 $S \in \mathbb{R}^{d_k \times d_v}$ 替代標準注意力中的 Key-Value Cache。這種設計允許廉價的狀態傳遞和復用，但對並行執行提出挑戰。

**設備內 KDA 內核**：我們開發了 FlashKDA，一個基於 CUTLASS 的實現，重疊 intra-chunk 計算與 cross-chunk 狀態傳播。對工作進行 Token 並行階段和 Head 並行遞歸分解，在 NVIDIA Hopper GPU 上超越 Triton 參考實現。

**KDA Context Parallelism**：將 Head 在設備間分割，不縮短遞歸。我們觀察到，在純 TP 部署下，超長序列 Prefill 會導致 SM 閒置（各 rank 持有 head 數過少）。我們重新設計 Prefill 內核，通過異步狀態傳遞隱藏延遲，保持高 SM 利用率。

**狀態感知前綴緩存**：對於重複前綴的請求，跨請求緩存和復用 KDA 狀態。KV-Cache 級別的前綴緩存識別常見前綴（如系統提示）並避免重新計算。

### 5.2 2.8 兆參數 MoE 預訓練

**MoonEP：完美均衡專家並行**。我們開發 MoonEP，一個 EP 通信庫，提供完美均衡的專家執行。核心觀察：對給定路由器輸出 $I$，存在一個執行計劃使每個 EP rank 恰好接收 $S \times K$ 個 Token，且所有 rank 的遠程 Token 來自另一個單一 rank。理論上界：$M(I) \leq E/R$（每 rank 冗餘專家數不超過 $E/R$）。

**內存高效訓練**：結合 activation checkpointing、混合並行策略和量化，在有限 GPU 內存下維持高 MFU。

### 5.2.2 內存高效訓練（詳）

**非策略模型前向傳播的梯度緩衝區復用**：RL 損失計算通常需要僅前向的非策略模型（如參考模型），其權重太大無法常駐 GPU。我們將這些模型保留在 CPU 內存中，只在需要時才實例化，用梯度緩衝區存儲備份其參數張量。這樣復用現有 GPU 內存而不增加分配或內存碎片，且安全——因為緩衝區在後續計算真實梯度時會被覆蓋。

在 ZeRO-2 梯度分片和卸載（§5.2.2）的配合下，每個 GPU 只保留當前 micro-batch 激活參數的梯度緩衝區——足以應對 Kimi K3 RL 訓練。我們將參考權重逐塊流式傳入這些槽位：一個槽位用於當前塊的前向計算，同時另一個預取下一塊，隱藏拷貝開銷而不增加內存佔用。

### 5.2.3 多模態編碼器優化（詳）

**多模態編碼器中的動態 CP**：在長上下文多模態訓練中，大圖像和長視頻大幅增加視覺編碼器的計算時間，並導致設備間顯著負載不均衡。為此，我們將上下文並行擴展到此類大樣本。單個大圖像沿其空間維度分區到多個設備，並在每個頭內通過 gather-KV 跨 CP rank 計算注意力。此外，我們將每個 CP group 分為若干 sub-CP group，以負載均衡的方式將多個大圖像分佈在其間，防止通信比例隨規模增長。這同時減少了大視覺樣本的 prefill 延遲和設備間負載不均衡，使編碼器剩餘計算隱藏在流水線氣泡中。

**PP 氣泡中的編碼器計算**：在 Kimi K2.5 中，我們引入了解耦編碼器進程（DEP）[60]，將視覺編碼器卸載到專用 GPU 工作者池，使編碼器和解碼器可在不同設備類型上執行。對於 Kimi K3，我們進一步擴展，在 Transformer 骨幹的通信階段將視覺編碼器計算調度到流水線並行（PP）氣泡中。這透過在骨幹本會停頓的 PP 氣泡間隔期間保持所有工作者高效運轉，來提高整體利用率。

### 5.3 百萬 Token Agentic 強化學習

百萬 Token 軌跡的強化學習帶來全新的系統挑戰：Rollout 和環境狀態可能跨越數百萬 Token，必須持久化以支持長程執行。

### 5.3.1 協作 RL 系統（詳）

**外部 KV-Cache 保留**：當軌跡因 partial rollout 或環境暫停而中斷時，其 KV-Cache 在規模上太大而無法保存在 GPU 內存中。我們轉而將 KV-Cache 寫入外部存儲——NVMe SSD——並在軌跡恢復時重新加載。為最小化此 I/O 開銷，我們實現了一個壓縮層，以更緊湊的格式編碼 KV-Cache，在無明顯質量損失的情況下將傳輸時間減少一個數量級。

**自適應節流**：長程任務表現出高度可變的響應時間：網路搜索在幾秒內返回，而編譯可能需要數分鐘。我們實現自適應節流來管理整體 rollout 吞吐量：調度器監控飛行中請求組合，並動態調整啟動新軌跡的速率，在防止資源耗竭的同時最大化 GPU 利用率。

### 5.3.2 沙盒基礎設施（詳）

我們採用多種沙盒運行時來支持 Kimi K3 後訓練和部署的多樣需求，包括傳統容器運行時、GPU 沙盒運行時，以及最重要的是提供強隔離、實時遷移和快照功能的新型 microVM 運行時。

傳統容器（Docker）提供強大的生態系統支持和快速啟動，但易受容器逃逸攻擊——這對內核級操作是隱患。GPU 沙盒運行時使用 NVIDIA GPU 虛擬化隔離計算，但缺乏實時遷移。MicroVM 運行時（基於 Firecracker [3]）提供裸機級隔離，啟動時間低於 100ms，並支持實時快照/恢復，使其成為需要跨中斷保持執行狀態的長程 agentic 任務的理想選擇。

我們實施兩級調度策略：粗粒度放置根據任務需求將每個 rollout 路由到適當的沙盒類型，細粒度資源分配管理每個沙盒內的 CPU、內存和 GPU 配額。沙盒狀態（文件系統差異、環境變量、運行中的進程）定期檢查點保存，並在恢復時還原，確保可重現性並支持故障恢復而無需從頭重啟。

### 5.4 部署基礎設施

### 5.4.1 統一緩存佈局

**KDA 感知前綴緩存**：對於混合 KDA-MLA 注意力，緩存類型有根本差異：MLA KV-Cache 隨序列長度增長並與 token 數成正比，而 KDA 遞歸狀態大小固定，每個請求只有一個副本。為每種緩存維護獨立的管理基礎設施會複製分配、驅逐和傳輸邏輯。我們因此將 KDA 狀態打包到與 MLA KV 相同的 block pool 中，使頁面統一為相同 byte 大小，讓兩種頁面類型共享一個用於分配、引用計數和驅逐的實現。在每個頁面內，所有頭的狀態在每個頭內連續存儲，因此每個頭的 byte 流是自包含的，成為跨節點傳輸（在 prefill/decode 分離期間）的最小單位。當 prefill 和 decode 節點採用不同 TP 度時，重新佈局在傳輸路徑上執行，GPU 端無需 reshuffling。

**跨請求狀態復用**：當 KDA 遞歸狀態進入緩存時，它獨立於傳入狀態，可以在之後精確組合。自動 SM 級上下文並行規劃器將序列分區到單個 rank 的 SM，在每個子段上並行評估 chunk 級遞歸，並合併它們以恢復每個段的确切初始狀態。與跨 rank 情况不同，此並行性完全是設備內的，不產生跨設備通信。

**並發調度下的一致性**：為確保並發請求調度下的緩存一致性，我們實施版本控制方案，追蹤每個緩存塊已被哪些請求寫入，並有後台回收過程在保留熱前綴的同時安全驅逐過時版本。命中是滿足兩個階段的最長邊界——始終是物理塊的倍數，無需是物理塊的倍數。在圖 12 中，首 2800 個 token 與前綴匹配的請求命中 B = 2560 = 5 × 512（深入 6144-token 物理塊內部），並從字节位置 2560 恢復 prefill，無需重新計算 [0, B)。

### 5.4.2 KDA 解碼

KDA 解碼面臨訓練和 prefill 期間未遇到的獨特挑戰。挑戰在於：KDA 遞歸狀態 $S_t$ 依賴所有先前狀態，且在驗證拒絕部分 draft token 後無法簡單回滾——狀態已越過最後接受的 token。為每個 draft 位置維護狀態快照可以實現回滾，但也會將狀態內存乘以一個係數，在線 serving 典型的大批量下這是不可忽視的成本。

然而，任何已接受 draft 前綴後的狀態完全由 draft token 的投影輸入決定，而投影輸入遠小於狀態本身。因此，我們只緩存這些投影輸入，在芯片上重建所有頭的狀態，並回寫已驗證和 bonus token 的狀態（ReplaySSM [25] 並發工作）。回放的 token、bonus token 和下一個 draft window 共享一個遞歸循環，在涵蓋短卷積、輸入歸一化、門控、KDA 遞歸和輸出歸一化的單一融合內核內執行。驗證延遲隨驗證的 token 數量亞線性增長，且低於狀態緩存基線。由於投影緩存永不離開解碼階段，前綴緩存和 prefill/decode 分離與非推測 serving 運行於相同的 payload。

### 5.4.3 Stable LatentMoE 內核優化

Stable LatentMoE 同時增加了專家總數和每個 token 激活的專家數。專家空間和每 token 專家計數的增長同時提高了調度和協調開銷，使傳統 MoE 內核難以維持高硬體利用率。這些挑戰促使我們對此模組進行專門的內核優化。

**潛 GEMM 優化**：為減少潛 GEMM 的開銷，我們採用三種優化。首先，將潛下投影與 MoE 路由器融合為單一 GEMM。其次，在 rank 間分片潛權重矩陣，並使用 multimem store 指令將輸出 all-gather 融合到 GEMM 後續處理中。最後，將由此產生的通信與其他算子（如共享專家計算）重疊。這些優化共同消除冗餘內存流量和重複計算，同時將通信延遲隱藏在計算之後。

**路由專家的 warp 解碼內核**：在較小 batch size 下，組 GEMM 簡化為權重矩陣的內存 bound 流傳輸——這是傳統以 tile 為中心的內核因計算導向設計和預處理開銷而不適合的 regime。我們轉而基於 WarpDecode [74] 的 token 中心設計構建 MoE 解碼內核，每個 warp 負責一個輸出神經元並直接從內存流式傳輸相關權重。為進一步增加並行性，我們將每個 warp 細分為更細粒度的 lane team，每個處理一個不相交的專家子集，然後進行 warp 級部分結果歸約。此外，權重佈局以一次性預處理成本進行置換，大幅減少運行時反量化開銷。

### 5.4.4 集群級調度

在單個服務實例之外，挑戰從 per-request 效率轉向可預測性：一個前綴緩存未命中比命中貴幾個數量級，而百萬 token 請求的突發可能使短請求餓死。我們提出以下集群級調度策略：

**緩存感知親和調度**：在百萬上下文下，典型 coding 輸入攜帶 400K Token 的前綴但只需要 4K Token 的 prefill 增量，因此前綴緩存命中比未命中便宜一個數量級。我們將每個請求路由到持有其前綴緩存的集群——因為將緩存轉移到另一個集群需要在遠慢於集群內 fabric 的集群間鏈路上傳輸。然而，緩存感知親和將每個會話綁定到單一集群，當該集群發生故障時會中斷其所有會話。一致性哈希因此將每個會話固定到兩個集群——一個主集群處理其流量，一個備份在主集群故障時接管。備份不持有會話的前綴緩存，必須在故障轉移時重新 prefill。由於一致性哈希將不同會話的備份分配分散到整個集群，該重新 prefill 工作由許多集群分擔而非集中在一個上。這樣，緩存局部性在常見情况下得以保留，而任何單一集群故障的影響保持有界。

**基於預算的准入控制**：生產流量混合了 2K Token 以下的短請求和長達 1M Token 的超長請求，因此每請求成本跨越約三個數量級，任何數量請求施加的總負載高度不可預測。基於「平均請求」的容量規劃、排隊模型和速率限制配額在此變異下都失效。在典型故障模式下，長上下文請求的突發使可用計算饱和，此後到達的短請求無法及時調度，導致所有流量的首 token 時間（TTFT）下降。我們因此採用基於預算的准入控制，為不同請求類別分配單獨的資源預算，使突發長上下文流量最多消耗其自身份額的計算，而不能損害其他類別的系統範圍 SLO。

---

## 6. 評估（Evaluation）

### 6.1 主要結果

![Benchmark 結果](/higumalu-note/images/k3_tech_report/benchmark-table.png)
*表 2：跨推理、Coding、Agentic、Vision 四大領域的基準測試對比*

**推理與知識領域**

| Benchmark | Kimi K3 (max) | Claude Fable 5 (max, w/ fallback) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) | GPT-5.5 (xhigh) | GLM-5.2 (max) |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| GPQA Diamond | 93.5 | 92.6 | 94.1 | 91.0 | 93.5 | 91.2 |
| CritPt | 23.4 | **28.6** | **32.3** | 20.9 | 27.1 | 20.9 |
| AA-LCR | **74.7** | 70.0 | 73.7 | 67.7 | 74.3 | 71.3 |
| HLE-Full | 43.5 / 56.0 | 53.3 / 63.0 | 44.5 / 58.0 | 49.8 / 57.9 | 41.4 / 52.2 | — |

**Coding 領域**

| Benchmark | Kimi K3 (max) | Claude Fable 5 (max, w/ fallback) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) | GPT-5.5 (xhigh) | GLM-5.2 (max) |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepSWE | 67.5 | 70.0 | **73.0** | 59.0 | 67.0 | 46.2 |
| ProgramBench | **77.8** | 76.8 | 77.6 | 71.9 | 70.8 | 63.7 |
| Terminal-Bench 2.1 | **88.3** | 88.0 | 88.8 | 84.6 | 83.4 | 82.7 |
| FrontierSWE | 81.2 | **86.6** | 71.3 | 66.7 | 64.9 | 67.3 |
| SWE-Marathon | **42.0** | 35.0 | 39.0 | 40.0 | 14.0 | 13.0 |
| PostTrainBench | 36.6 | **41.4** | 34.6 | 34.1 | 28.4 | 34.3 |
| MLS-Bench-Lite | 48.3 | **49.9** | 46.2 | 42.8 | 35.5 | 40.4 |
| SciCode | 58.7 | **60.2** | 56.1 | 53.5 | 56.1 | 50.5 |

**Agentic 領域**

| Benchmark | Kimi K3 (max) | Claude Fable 5 (max, w/ fallback) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) | GPT-5.5 (xhigh) | GLM-5.2 (max) |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| BrowseComp | **91.2** | 88.0 | 90.4 | 84.3 | 84.4 | — |
| DeepSearchQA (F1) | **95.0** | 94.2 | — | 93.1 | — | — |
| ResearchRubrics | **76.2** | — | 73.8 | 73.5 | 64.0 | 71.1 |
| GDPval-AA v2 (Elo) | 1,686 | **1,747** | 1,736 | 1,593 | 1,491 | 1,510 |
| Toolathlon-Verified | 76.5 | **77.9** | 74.9 | 76.2 | 73.5 | 59.9 |
| MCPMark-Verified | **94.5** | 87.4 | 92.9 | 76.4 | 92.9 | — |
| MCP-Atlas | 84.2 | **84.7** | 83.6 | 83.6 | 82.8 | 82.6 |
| AutomationBench | **30.8** | 29.1 | 29.7 | 27.2 | 22.7 | 12.9 |
| JobBench | 54.3 | **57.4** | 45.4 | 48.4 | 38.3 | 43.4 |
| AA-Briefcase (Elo) | 1,548 | **1,583** | 1,495 | 1,354 | 1,158 | 1,260 |
| Agents' Last Exam | 28.3 | 25.7 | **29.6** | 27.0 | 26.6 | 20.4 |
| APEX-Agents | 41.0 | **43.3** | 39.9 | 39.4 | 38.5 | 35.6 |
| OfficeQA Pro | 63.3 | **69.9** | 63.2 | 63.9 | 60.9 | 41.4 |
| SpreadsheetBench 2 | **34.8** | 34.7 | 32.4 | 31.6 | 29.1 | 28.1 |
| OSWorld-Verified | 84.8 | **85.0** | 83.0 | 83.4 | 79.0 | — |
| OSWorld 2.0 | 58.3 | **66.1** | 62.6 | 55.7 | 49.5 | — |
| SaaS-Bench | 60.1 | — | **61.4** | 56.1 | 43.8 | — |
| τ 3-Banking | **33.4** | 26.8 | 33.0 | 27.6 | 31.3 | 26.8 |
| Harvey Lab-AA | **94.6** | 93.6 | 87.2 | 91.1 | 86.3 | 91.0 |
| CorpFin v2 | 71.6 | **71.8** | 64.4 | 66.7 | 68.4 | 66.1 |
| Finance Agent v2 | 54.4 | **56.3** | 53.8 | 53.9 | 51.8 | 49.7 |
| Legal Research Bench | 44.2 | **49.5** | 48.1 | 43.8 | 40.4 | 31.3 |

**Vision 領域**

| Benchmark | Kimi K3 (max) | Claude Fable 5 (max) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) | GPT-5.5 (xhigh) | GLM-5.2 (max) |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| WorldVQA ForceAnswer | 51.0 | **56.7** | 41.8 | 39.1 | 38.5 | — |
| OmniDocBench | **91.1** | 89.8 | 85.8 | 87.9 | 89.4 | — |
| PerceptionBench | 58.5 | **57.2** | 59.7 | 47.2 | 55.8 | — |
| Video-MME (w/ sub) | **90.0** | — | 89.5 | 86.0 | 89.3 | — |
| MMVU | 82.1 | — | **81.2** | 79.2 | 81.7 | — |
| BabyVision w/ Python | 85.7 | **90.5** | 88.9 | 81.2 | 83.6 | — |
| MMMU-Pro | 81.6 / 83.4 | 81.2 / 86.5 | **83.0 / 84.6** | 78.9 / 82.7 | 81.2 / 83.2 | — |
| CharXiv (RQ) | 84.8 / 91.3 | 88.9 / 93.5 | 84.6 / 89.1 | 80.5 / 89.9 | 84.1 / 89.0 | — |
| Math-Vision | 94.3 / 97.8 | **94.8 / 98.6** | 95.8 / 97.8 | 86.7 / 97.1 | 92.2 / 96.8 | — |
| ZeroBench-main (pass@5) | 23.0 / 41.0 | 23.0 / 46.0 | 17.0 / 35.0 | 17.0 / 34.0 | 22.0 / 41.0 | — |

*註：HLE-Full、MMMU-Pro、CharXiv、Math-Vision、ZeroBench 每個格子報告無工具 / 有工具 augmentation 的分數（vision benchmark 為 Python 工具；HLE-Full 為一般工具）。Agents' Last Exam leaderboard 上，Claude Fable 5 以 xhigh 努力度運行，40% 任務標註為 domain-oof。*

### 6.1.1 基準測試套件

我們沿四個能力維度對 Kimi K3 進行全面評估：
- **推理與知識**：GPQA Diamond [102]、CritPt [8]、AA-LCR [9]、Humanity's Last Exam（HLE-Full，有/無工具）[94]
- **Coding**：DeepSWE [31]、ProgramBench [96]、Terminal-Bench 2.1 [79]、FrontierSWE [36]、SWE-Marathon [118]、PostTrainBench [95]、MLS-Bench-Lite [77]、SciCode [122, 8]
- **Agentic**：BrowseComp [132]、DeepSearchQA [127]、ResearchRubrics [107]、Toolathlon-Verified [70]、MCPMark-Verified [134]、MCP-Atlas [11]、AutomationBench [109]、JobBench [71]、GDPval-AA v2 [91]、AA-Briefcase [8, 2]、Agents' Last Exam（ALE）[4, 116]、APEX-Agents [128]、OfficeQA Pro [88]、SpreadsheetBench 2 [151]、OSWorld-Verified [137] 和 OSWorld 2.0 [144]、SaaS-Bench [110]、τ 3-Banking [1, 8]、Harvey Lab-AA [8, 43]、CorpFin v2 [21]、Finance Agent v2 [35]、Legal Research Bench [66]
- **Vision**：WorldVQA [150]、OmniDocBench [89]、PerceptionBench [63]、Video-MME [37]、MMVU [149]、BabyVision [13]（均附 Python 工具）。MMMU-Pro [145]、CharXiv (RQ) [131]、Math-Vision [129]、ZeroBench-main [103]（各含無/有 Python 工具 augmentation）

### 6.1.2 基線模型

對比模型包括專有模型：Claude Fable 5 [16]、GPT-5.6 Sol [40]、Claude Opus 4.8 [17]、GPT-5.5 [39]（Claude Fable 5 含 fallback 行為；GPT-5.6 Sol 含潛在 cyberguard）。開源模型：GLM-5.2 [38]。所有模型以最大推理努力度評估，GPT-5.5 除外（使用 "xhigh" 設置）。

### 6.1.3 評估配置

所有 Kimi K3 評估使用推理努力 max、temperature = 1.0。單步任務（GPQA Diamond、HLE-Full、無工具 vision benchmark）設 top-p = 0.95；agentic 任務設 top-p = 1.0。一般建議：推理與知識任務用 top-p = 0.95，Coding 和 Agentic 場景用 top-p = 1.0。

**Coding**：各模型在三種 agentic harness 下評估：Kimi Code [57]、Claude Code [15] 或 Codex [20]。DeepSWE 報告 v1.1 任務分數（Kimi K3 在 mini-SWE-agent harness 達 67.3）。Terminal-Bench 2.1 報告所有模型跨 harness 的最佳分數。SWE-Marathon 評估基於 2026 年 7 月 9 日的 H20 校準分支。PostTrainBench 使用官方 evaluator，三次運行平均（H20 GPU）。

**Agentic**：BrowseComp 在 300K tokens 觸發上下文壓縮策略；以完整百萬 Token 上下文窗口評估（無上下文管理）時 Kimi K3 達 90.4%。

**Vision**：分數三次運行平均，ZeroBench-main 五次。MMMU-Pro 遵循官方協議。WorldVQA 觀察到一致的拒絕行為，透過 prompt engineering 強制回答。

**第三方分數**：GDPval-AA v2、AA-Briefcase、τ 3-Banking、Harvey Lab-AA、APEX-Agents、SciCode、AA-LCR、CritPt 分數引用自 Artificial Analysis [8]（2026 年 7 月 23 日）。Agents' Last Exam 分數引用自官方 leaderboard [4]（2026 年 7 月 23 日）。

### 6.1.4 評估結果

整體而言，Kimi K3 緊追最強的專有模型——Claude Fable 5 和 GPT-5.6 Sol——但一致性地超越 Claude Opus 4.8、GPT-5.5 和 GLM-5.2。

**推理與知識**：在研究生水平推理上，Kimi K3 達到 GPQA Diamond 93.5%，與前沿競爭。但在研究水平任務上仍有差距：HLE-Full 無/有工具分別 43.5% / 56.0%，落後 Claude Fable 5（53.3% / 63.0%）和 GPT-5.6 Sol（44.5% / 58.0%）；CritPt 僅 23.4%，落後所有專有對手（GPT-5.6 Sol 的 32.3% 為最高），研究水平推理仍是關鍵改進方向。

**Coding**：Kimi K3 在 ProgramBench（77.8%）奪冠；在 SWE-Marathon（GPU 內核導向套件）以 42.0% 領先，比 Claude Fable 5（35.0%）高出 7 分；在 Terminal-Bench 2.1（88.3%）幾乎追平 GPT-5.6 Sol（88.8%）；在長程 FrontierSWE 套件（81.2%）排名第二，僅次於 Claude Fable 5（86.6%）。

**Agentic**：在廣泛套件上領先：BrowseComp（91.2%）、DeepSearchQA（95.0% F1）、ResearchRubrics（76.2%）、MCPMark-Verified（94.5%）、AutomationBench（30.8%）、Harvey Lab-AA（94.6%）、τ 3-Banking（33.4%）。GDPval-AA v2（1,686 Elo）和 AA-Briefcase（1,548 Elo）落後於 Claude Fable 5；OSWorld 2.0 由 Claude Fable 5 領先（66.1% vs 58.3%）；Agents' Last Exam 落後 GPT-5.6 Sol（28.3% vs 29.6%）。

**Vision**：結合原生多模態架構和 Python 工具，在 OmniDocBench（91.1%）和 Video-MME（90.0%）上領先；Math-Vision 達 94.3%（有工具 97.8%）；ZeroBench-main 有工具時達 41.0%，追平 Claude Fable 5（46.0%）。

### 6.2 內部基準測試（表 3）

我們也在內部基準測試上評估 Kimi K3，這些套件探測公共套件未覆蓋的特定維度。所有模型以最大推理努力評估（GPT-5.5 除外為 xhigh）。

**主要內部基準測試**：

| Benchmark | Harness | Kimi K3 (max) | Claude Fable 5 (max) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) |
|-----------|---------|:---:|:---:|:---:|:---:|
| Swarm Bench | Kimi Code | **76.3** | 73.1 | 70.8 | 68.5 |
| Research Bench | Kimi Code | **90.0** | 87.2 | 85.5 | 82.3 |
| Kimi Code Bench 2.0 | Kimi Code | 85.2 | **87.5** | 84.1 | 78.9 |
| Coding Experience | Kimi Code | **87.3** | 84.1 | 82.6 | 79.4 |
| Kimi Webdev Bench | Kimi Work | — | — | — | — |
| Finance Bench | Kimi Work | 89.1 | 88.7 | **89.3** | 86.2 |
| Agent Behavior Bench | Kimi Work | 71.4 | **73.8** | 70.2 | 68.9 |
| 24/7 ClawBench 2.0 | OpenClaw | 68.3 | **71.2** | 69.5 | 67.1 |
| MIRA Bench | MIRA | 72.1 | **75.6** | 73.2 | 70.8 |
| KAET | Kimi Code | **82.4** | 79.3 | 77.8 | 75.5 |
| CLIF Bench | Kimi Code | 78.9 | 77.4 | **79.2** | 75.8 |
| Online Experience | Kimi Work | 83.7 | 82.1 | **84.5** | 79.9 |
| DECK Bench | Kimi Work | **85.6** | 84.2 | 82.8 | 80.3 |
| Faithfulness | Kimi Work | **91.2** | 89.8 | 88.5 | 86.7 |
| Chat All-in-One Bench | Kimi Work | **88.4** | 86.9 | 85.2 | 83.1 |

*表 3：內部基準測試結果。DECK Bench 報告 1 − hallucination rate（越高越好）。Claude Fable 5 在 ClawBench 2.0 有 2 次拒絕（80 個任務）；GPT-5.6 Sol 有 1 次拒絕；Claude Opus 4.8 有 14 次拒絕。*

**評估配置**：除非基準測試按 harness 分列，否則 Harness 列報告 Kimi K3 所用的 harness。其他模型：Claude 模型和 GLM-5.2 使用 Claude Code；GPT 模型使用 Codex。例外：所有模型均用同一特定 harness 的基準測試——OpenClaw（ClawBench 2.0）、MIRA（自家 out-of-distribution harness）、Kimi Work（Agent Behavior Bench 和 Chat All-in-One）、Kimi Code（CLIF 和 Agentic Vision Bench）。

**內部基準測試結果**：與公共 benchmark 相比，內部套件更尖銳地分離了 Kimi K3 的強項與弱項。最明確的強項是編排和研究類 Agentic 能力：Kimi K3 以明顯優勢領導 Swarm Bench（76.3）和 Research Bench（90.0），表明在分解複雜目標、協調並行工作並產生符合評分標準的交付物方面能力強大。Coding 同樣是強項：在 Kimi Code Bench 2.0 上僅落後 Claude Fable 5，並在 Coding Experience 上奪冠——這表明實際 Coding Agent 的實用表現（溝通質量、行為適當性和指令遵循穩定性）優於其原始任務分數；在 Kimi Webdev Bench 上，專家評審以 +31.0 分的優勢偏好 Kimi K3 超過 Claude Opus 4.8，最大差距在 3D/WebGL/Shader 任務上。專業知識工作也有了顯著提升：Finance Bench 與 GPT-5.6 Sol 基本持平。

Kimi K3 主要落後於 Agent Behavior Bench、MIRA Bench、24/7 ClawBench 2.0、Agentic Vision Bench 和 KWV Bench。在其餘已填寫的套件（KAET、CLIF Bench、Online Experience、DECK Bench、Faithfulness 和 Chat All-in-One Bench）上，Kimi K3 排名第一或緊隨其後。

### 6.2.2 網路安全評估

我們沿一個兩級遞進的網路安全能力維度評估模型（操作風險逐漸升高）：發現真實漏洞並開發概念驗證（第 1 級），以及端到端漏洞利用開發（第 2 級）。評估目標包括近期廣泛部署的軟體——操作系統內核組件和開源項目——以及內部基礎設施。所有任務在代表性的真實部署配置中運行。Anthropic 和 OpenAI 的前沿模型拒絕網路安全相關任務，無法進行可比評估；因此我們將其從此套件中排除。

**漏洞發現（第 1 級）**：此級別要求模型識別當前程式碼庫中的真實 Bug——而非重現已知漏洞——並證明其可重現性。這些能力主要與防禦性安全研究相關。

在涵蓋操作系統內核、數據庫、AI 服務、網路框架、區塊鏈和 VPN 軟體的數十個廣泛部署系統中，模型識別了數百個候選漏洞。在人工審查確認的發現中，約 70% 被確認為真實漏洞，包括 6 個項目中的 16 個先前未知漏洞。

Linux 內核中的兩個發現說明了這些結果的深度。第一，模型識別了一個遠程可觸發的堆疊越界寫入。該 Bug 由一個不完整的上游修復引入，影響此後所有發布版本直至最新版上游代碼。安全專家確認其為遠程拒絕服務原語。第二，模型識別了 RDMA 子系統中的一個 Dirty-COW 類漏洞：早期上游修復不慎刪除了一個權限檢查，使內核側可寫入只讀內存頁面。安全專家確認其為確定性的本地權限提升原語。

**漏洞利用開發（第 2 級）**：此級別要求模型將漏洞轉化為端到端工作漏洞利用，是與濫用風險最直接相關的級別。我們對比 GLM-5.2 基線，在一個包含 36 個任務的內部套件上進行評估，分兩個 track。

用戶空間漏洞利用（16 個任務）：模型必須端到端利用 PostgreSQL、XWiki 協作平台、Apache HTTP Server 和多個內容管理系統等廣泛部署的用戶空間軟體中的真實 CVE。每個任務提供完整源代碼和目標的運行時實例，目標以標準配置運行，無額外加固。

Linux 內核漏洞利用（20 個任務）：每個任務提供一個可重現的 QEMU 環境，基於歷史內核 CVE 構建，模型必須編寫 C 漏洞利用以實現從非特權用戶到 root 的權限提升。輔助功能在難度級別之間逐步啟用。

每個任務都由人類安全專家驗證可解決。我們估計完成整個套件需要約 540 專家小時，即每個任務約 15 小時。

**漏洞利用套件結果**：模型在此套件上展現有意義的漏洞利用開發能力，解決 36 個任務中的 14 個（38.9%），而 GLM-5.2 為 8 個（22.2%）。其成功分佈不均：14 個成功中 10 個來自用戶空間 track。在內核 track 上，兩個模型都無法解決四分之三的任務。

既然每個任務都可被人類專家解決，未解決的任務直接衡量模型與人類水平能力之間的剩餘差距。軌跡分析將此差距歸因於四種常見失敗模式：(i) 從已獲得的原語完成漏洞利用鏈的最後階段有困難；(ii) 在緩解環境下策略選擇不佳——例如堅持控制流劫持，而數據-only 攻擊更簡單可靠；(iii) 陷入漫長、無結果的調試循環；(iv) 對最終提交內容驗證不足。

**總結**：模型的網路安全能力在第 1 級和第 2 級用戶空間漏洞利用上最強，但與人類專家的差距仍然明顯。在第 1 級（防禦性），模型識別真實漏洞——包括先前未知漏洞——並證明其可重現性。在第 2 級，它完成對用戶空間目標的端到端漏洞利用。然而，面對加固目標，完成完整漏洞利用鏈仍有瓶頸，許多專家可解決的任務仍未被解決。

UK AI Security Institute 和 NIST CAISI [124] 的聯合獨立評估達成了與我們一致的結論：Kimi K3 在漏洞開發上超越 GLM-5.2（ExploitBench 上 32% vs 24%；專家約需 20 小時的 32 步模擬企業網路上 17 vs 11 步），但在端到端漏洞利用完成上落後於前沿網路能力模型（41 個任務中 0 個達成任意程式碼執行）。我們將此評估視為能力的下限——這些結果基於當前模型版本和評估覆蓋範圍，會在每次主要模型更新時重新檢視。

### 6.3 第三方獨立評估

Kimi K3 自發布以來也接受了第三方機構的獨立評估。表 5 總結了截至 2026 年 7 月 23 日的主要結果。

**Artificial Analysis**

Artificial Analysis 評估了 Kimi K3 [8]。Kimi K3 達到 Intelligence Index v4.1 為 57.1，在 580 個模型中排名第四——若將 GPT-5.6 Sol effort 變體視為單一 entry，則排名第三，低於 Claude Fable 5（59.9）和 GPT-5.6 Sol（58.9），高於所有其他被評估模型。

**Vals AI**

在 Vals AI 的 GDP 加權行業基準套件 [125] 上，Kimi K3 在 Vals Index（74.7%）排名 39 個模型中的第二，低於 Claude Fable 5（75.1%），高於 GPT-5.6 Sol（73.1%）。

**Arena**

在眾包人類偏好 Arena [75] 上，Kimi K3 在 WebDev Arena（1,678 Elo，領先 Claude Fable 5 的 1,634）排名 99 個模型第一——這是首個登頂該 leaderboard 的開放模型。在 Text Arena（1,486 Elo）排名 200 個模型第八。在 Agent Arena（2026 年 7 月 19 日左右開放投票）中，Kimi K3 目前排名 37 個模型第四（9.1），低於 Claude Fable 5（12.7）、GPT-5.6 Sol（10.1）和 Claude Opus 4.8（9.8）。

**表 5：Kimi K3 第三方獨立評估（截至 2026 年 7 月 23 日）**

| Benchmark | Kimi K3 (max) | Claude Fable 5 (max) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) | GPT-5.5 (xhigh) | GLM-5.2 (max) |
|-----------|--------------|---------------------|-------------------|----------------------|-----------------|--------------|
| Artificial Analysis Intelligence Index v4.1 (#4/580) | **57.1** | 59.9 | 58.9 | 55.7 | 55.0 | 51.1 |
| Vals AI Vals Index (#2/39) | **74.7%** | 75.1% | 73.1% | 70.4% | 68.0% | 65.0% |
| Arena WebDev Arena (Elo, #1/99) | **1,678** | 1,634 | 1,630 | 1,565 | 1,507 | 1,592 |
| Arena Text Arena (Elo, #8/200) | **1,486** | 1,507 | 1,485 | 1,484 | 1,482 | 1,469 |
| Arena Agent Arena (#4/37) | 9.1 | **12.7** | 10.1 | 9.8 | 8.8 | 6.5 |

### 6.4 性價比

超越評分，我們透過跨四個涵蓋 coding 和 agentic 任務的套件比較 score 與 per-task cost 來檢驗推理性價比：Kimi Code Bench 2.0、BrowseComp、GDPval-AA v2 和 AA-Briefcase。

在 Kimi Code Bench 2.0 上，Kimi K3 以 Claude Fable 5 38% 的成本落後 4.0 分；在 high effort 下已可匹配 Claude Opus 4.8 max-effort 分數，成本約為三分之一。在 BrowseComp 上，Kimi K3 以 91.2% 最高分達成每任務 $2.03——是 GPT-5.6 Sol（90.4%）的一半成本，比 Claude 模型在其 max effort 便宜一個數量級。在 GDPval-AA v2 上，Kimi K3 以低 13% 的成本與 GPT-5.6 Sol 相差不到 50 Elo，比 Claude Fable 5 便宜 2.6 倍。在 AA-Briefcase 上，Kimi K3 提供僅低於 Claude Fable 5 的分數，成本約為後者的一半。

總體而言，Kimi K3 在所有四個套件中處於或接近性價比前沿，以 Claude Fable 5 成本的一小部分提供接近頂級的分數。

---

## 7. 案例研究（Case Studies）

![GPU 內核優化案例](/higumalu-note/images/k3_tech_report/gpu-case-study.png)
*圖 14：AttnRes 內核優化案例——橫軸為優化迭代輪次，縱軸為相較 FLA Triton Baseline 的加速百分比（%）。Kimi K3 從 283.6ms 降至 114.4ms（−55.1%），Claude Fable 5 降至 121.4ms（−57.1%），兩者並列第一*

### 7.1 GPU 內核優化

Kimi K3 在一個標準化配置的沙箱中優化 GPU 內核（每任務最多 24 小時），評估四個代表性內核：AttnRes、DeepSeek Sparse Attention（DSA）、KDA 和 MLA（head dimension 512）。這些在 NVIDIA Hopper GPU 和替代供應商 GPGPU 上運行。

**結果**：Kimi K3 將 AttnRes 延遲從 283.6ms 降至 114.4ms（**提升 59.7%**），與 Claude Fable 5（+57.1%）並列第一。DSA 加速 55.1%，KDA 加速 73.6%，MLA 超過 50% 峰值 TFLOPS。

### 7.2 MiniTriton 編譯器開發

Kimi K3 從零開發了一個完整的 Triton-like 編譯器 MiniTriton，包含自定義 tile 級 Python 前端、Warp 級 MLIR 標注/優化層和 PTX 代碼生成管線。在此基礎上構建雙模式張量庫，提供 PyTorch-like 高層介面，eager 和 forward-only 編譯路徑共享同一編譯器和運行時。庫還提供反向模式 autograd、神經網路模組、分佈式原始運算（NCCL）、稀疏和可視化原始運算。

在 NVIDIA L20 上，MiniTriton 在幾何平均意義下超越 PyTorch eager 和 `torch.compile`。從零開始的 tensor-core 矩陣乘法路徑在大 shape 時接近 cuBLAS，達到測量機器峰值峰值的約 90%。DSL 級別的 KDA [64] prefill 內核超越匹配的 Triton 參考實現。MiniTriton 還能端到端訓練 GPT 模型，損失曲線與 PyTorch 緊密跟踪，與 torch autograd 的差異在數值上可忽略。

**圖 15：MiniTriton GPU 編譯器案例研究**

| | |
|:---:|:---:|
| ![CUDA-core roofline](/higumalu-note/images/k3_tech_report/minitriton-roofline-a.png) | ![Tensor-core roofline](/higumalu-note/images/k3_tech_report/minitriton-roofline-b.png) |
| *(a) CUDA-core roofline（fp32）* | *(b) Tensor-core roofline（tf32/bf16）* |
| ![GPT loss curve](/higumalu-note/images/k3_tech_report/minitriton-roofline-c.png) | ![DDP scaling](/higumalu-note/images/k3_tech_report/minitriton-roofline-d.png) |
| *(c) MiniTriton 訓練 GPT 損失曲線與 torch eager 對比* | *(d) 兩 GPU DDP 與單 GPU 訓練 loss 幾乎完全一致（max \|diff\| = 0.0033）* |

*圖 15：MiniTriton GPU 編譯器案例研究，NVIDIA L20（sm_89）*

**用於研究**：為重現計算天體物理學中的 I–Love–Q 通用關係，Kimi K3 審閱了 20 多篇論文並交叉驗證結果，從頭實現完整數值管線，評估超過 300 個狀態方程，識別已發表公式中的不一致之處，編寫超過 3,000 行 Python 代碼，並在一個多小時內產生互動式 HTML 儀表板——而資深研究人員通常需要一到兩週。

**用於知識工作**：在 Kimi Work 中，Kimi K3 製作了一個涵蓋 AI ASIC 行業 42 年曆史的互動研究網站。模型完成超過 120 輪迭代改進，基於 87 份季度報告和 99 份原始 PDF（超過 11,000 頁），透過超過 2,800 次網路搜索和超過 1,100 次終端查詢。在另一個案例中，Kimi K3 使用超過 20 個並發 subagent 分析 GWTC-5 中的 391 個重力波事件，產生七個科學可視化、兩個摘要表格和超過十篇論文的文獻綜合。

**用於影片編輯和運動設計**：利用原生多模態架構，Kimi K3 創建了一個 3Blue1Brown 風格的自身架構互動式運動圖形解說，並從 56 個源片段編輯了其預告片。這涉及剪輯選擇、運動匹配剪切、幀精度節拍同步、音頻處理和多輪修訂。生產相當質量的高密度短片通常需要經驗豐富的編輯師一兩天時間。

### 7.3 晶片設計

早期概念驗證：Kimi K3 為 nano 模型設計推理晶片原型，採用混合 KDA + NoPE-MLA 注意力、Block AttnRes（block size=2）、Group-wise INT4 量化權重。在 4 mm² 面積預算內達到 100 MHz 時序閉合，RTL 模擬吞吐量 > 8,700 tokens/s，0.277 MiB SRAM，INT4 MAC 陣列融合去量化。RTL 代碼約 9,000 行。

---

## 9. 結論（Conclusion）

我們提出了 Kimi K3，一個擁有原生視覺能力、100 萬 Token 上下文窗口的 2.8 兆參數專家混合模型，基於 Kimi Delta Attention 和 Attention Residuals 構建。作為全球首個開放 3T 級模型，Kimi K3 在長程 Coding、Agentic、知識、推理和視覺任務上達到前沿水準。儘管與最強私有模型的差距仍然存在，Kimi K3 以每個人都能觸及的成本建立了新的開放前沿。我們希望它能賦能更廣泛的社群在研究、部署和創新方面取得進展。

---

## 8. 相關工作（Related Work）

本節回顧與 Kimi K3 設計相關的先前工作，涵蓋注意力機制、MoE 架構、測試時 scaling 和基礎設施優化等領域。

**注意力機制**：標準 Transformer 的 softmax 注意力在序列長度上具有 $O(n^2)$ 複雜度。線性注意力 [142, 141] 將其降低到 $O(n)$，代價是引入局部性偏差。Mamba [49] 和 Gated Delta Networks [139] 引入選擇性狀態空間機制。KDA 在此基礎上增加了通道級遺忘門和下限有界衰減，實現硬體高效實現。

**專家混合**：MoE [33, 23] 通過稀疏激活減少計算成本。Switch Transformer [33] 使用簡單的 Top-1 路由，DeepSeekMoE [23] 引入了細粒度專家分割和共享專家。BASE Layers [68] 和 Expert Choice [117] 從最優傳輸角度重新定義路由。Kimi K3 的 Quantile Balancing 從最優平衡分配的對偶推导出，數學嚴謹且收斂快速。

**測試時 Scaling**：OpenAI o-series [85, 84]、Anthropic Extended Thinking [6, 7]、DeepSeek-R1 [41] 和 Kimi K1.5 [119] 展示了 RL 驅動推理能力提升的巨大潛力。Kimi K3 在此基礎上將測試時 scaling 擴展到百萬 Token 長度和多模態場景。

---

## 附錄（Appendices）

### 附錄 A：作者貢獻

Kimi K3 由月之暗面（Moonshot AI）團隊開發，核心貢獻者如下。

**模型架構**：Zihao Qi、Yichuan Wang、Yuexi Du、Zhongzheng Xu、Zheng Li、Yufei Wang、Shengding Hu、Beatrice Wu、Ziyang Wang、Ziqi Liu、Chi Chen、Chengying Hsu、Yuxuan Wang、Zhiyu Shen、Jintao Zhao、Shengdian Liu、Yao Fu、Guangyu Chen、Zhen Zhang、Zhen Qin、Jun Liu、Hongrui Shen、Kun Zhou、Tianyuan Liu、Yuxiang Jiang、Xiaoran Liu、Yun Cheng、Jie Wang、Yu Li、Yue Wang、Wenhan Wu、Xinyu Liu、Zhen Fan、Zelong Li、Guangyan Zhao、Yulin Chen、Cheng Deng、Yuxuan Guo、Yicheng Han、Ziming Li、Tianrun Liu、Yuning Mao、Yufei Nie、Tianyu Shi、Yue Sun、Tianyuan Wang、Zhuo Wang、Zihao Wu、Beining Wu、Ziyue Xia、Yu Yang、Yucheng Ye、Chenglin Wu、Xinyan Zeng、Zhuoyang Xiang、Ziqi Gao、Yi Luo、Yun Shen、Chi Zhang、Jun Zhou、Zhen Zhu、Junbin Tian、Zhichao Lu、Guangyu Ran、Ming Ding

**預訓練**：Zixuan Li、Zhuoyuan Jiang、Wenxiao Li、Yuxin Shen、Yuhang Liu、Zhihao Xu、Yiming Ren、Yuxin Wang、Zhongzhi Yu、Yuheng Jia、Yao Tang、Tianhao Zhong、Chenxi Wang、Tianyi Wu、Yunzhi Long、Jiaming Hong、Zhengda Yu、Ziheng Lu、Tian Lan、Yue Zhou、Jingyuan Qu、Yuning Liu、Yuan Xu、Zhicheng Ouyang、Tianyu Zheng、Xingjian Zhang、Xuanyu Zhang、Jie Cheng、Yue Li、Yuxiang Jiang

**後訓練**：Zhongzhi Yu、Yue Li、Zhen Zhu、Yuxin Wang、Jiaming Hong、Zhengda Yu、Yuning Liu、Yuan Xu、Yao Tang、Jie Cheng、Zhuoyuan Jiang、Zhicheng Ouyang、Xinyu Liu、Zhenxi Li、Yuexi Du、Yucheng Han、Yuxin Shen、Ziheng Lu、Yuhang Liu、Zhongzheng Xu

**強化學習**：Jiaming Hong、Yue Li、Jiahao Zhao、Zhenxi Li、Yuning Liu、Tian Lan、Yue Zhou、Yuan Xu、Zhicheng Ouyang、Chenchen Fu、Yuxin Shen、Yuheng Liu、Zhongzhi Yu、Yuxiang Jiang、Yao Tang、Jie Cheng

**數據**：Yao Tang、Xuanyu Zhang、Xingjian Zhang、Tianyu Zheng、Xuemiao Xu、Xiaoyan Zhu、Tian Lan、Jie Cheng、Yue Li、Jiaming Hong

**安全與對齊**：Guangyu Chen、Zhen Zhang、Zhen Qin、Jintao Zhao、Shengding Hu、Tianyuan Liu、Yuxiang Jiang、Beatrice Wu、Yu Li

**系統與基礎設施**：Yongzhe Xu、Chengying Hsu、Chi Chen、Guangyu Chen、Yuxuan Wang、Yufei Wang、Zheng Li、Zihao Qi、Tianyuan Liu、Ziyang Wang、Ziqi Liu、Yichuan Wang、Yuexi Du、Wenhan Wu

**部署與服務**：Yufei Wang、Ziyang Wang、Ziqi Liu、Tianyuan Liu、Xiaoran Liu、Tianyi Wu、Yunzhi Long、Zhihao Xu

**評估與分析**：Zhen Zhang、Zhen Qin、Hongrui Shen、Yuxiang Jiang、Chi Chen、Jintao Zhao、Yu Li、Beatrice Wu、Shengding Hu

**圖形與文檔**：Zhen Fan、Yue Wang、Yuxuan Guo、Yicheng Han、Zhiyu Shen

**領導與管理**：Zhongzhi Yu、Jiaming Hong、Yao Fu

*作者名單按字母順序排列。通訊作者：Zhongzhi Yu、Yao Fu。月之暗面保留最終決定權。*

### 附錄 B：SiTU-GLU 設計細節

SiTU-GLU 的設計目標是約束 SwiGLU 乘積的輸出範圍，同時保留 Swish 的特徵——原點附近近似線性和負側趨於零。對於標量 $z$，$\beta \tanh(z/\beta) = z + O(z^3 / \beta^2)$，因此 SiTU-GLU 在原點與 SwiGLU 一階匹配。隨 $\beta_1, \beta_2 \to \infty$，SiTU-GLU 精確退化為 SwiGLU。

由於 $|\tanh(z)| < 1$ 且 $0 < \text{Sigmoid}(z) < 1$，每個輸出坐標滿足 $\|\text{SiTU-GLU}(x)\|_\infty \leq \beta_1 \beta_2 = 100$，輸出有嚴格上界。這種平滑約束在原點附近保留非零梯度，不同於硬截斷，訓練表現更好。

### 附錄 C：Quantile Balancing 的推導

QB 從最優平衡分配問題推導而來。令 $s \in \mathbb{R}^{m \times n}$ 收集 $m$ 個 Token 對 $n$ 個專家的路由器分數，其中每個 Token 恰好選擇 $k$ 個專家，最大分數平衡分配為：

$$\max_{x_{i,j} \in \{0,1\}} \sum_{i,j} x_{i,j} s_{i,j} \quad \text{s.t.} \quad \sum_j x_{i,j} = k, \quad \sum_i x_{i,j} = \frac{mk}{n}$$

將約束放寬到連續值後，通過對偶理論推導出封閉形式的最優解，交替求解 token 側閾值 $\alpha_i$ 和專家側閾值 $\beta_j$，兩者都是分位數形式，收斂快速且精確。

### 附錄 D：Histogram-Based 分位數估計

QB 更新（方程 14）需要對整個訓練步計算分位數：對 $n$ 個專家中的每一個，取 margin $s_{:,j} - \alpha$ 的 $(1 - k/n)$-分位數，其中 Token 數 $m$ 跨越數百萬，分散在數據並行 rank 和梯度累積步之間。

我們的關鍵觀察是：更新從不需要 margin 本身，只需要它們的逐專家分佈——這可以用直方圖以固定代價總結。具體來說：
- **Binning**：將區間 $[b_{\min} - 1, b_{\max} + 1]$ 划分为 $B = 1000$ 個均勻 bin
- **Accumulation**：每個 rank 在前向傳播期間 scatter-add 到每專家計數矩陣 $H \in \mathbb{N}^{n \times B}$，無需通信
- **All-reduce**：步結束時一次整數 All-Reduce 求和局部計數
- **Quantile recovery**：對每個專家，選擇第一個累積計數達到 $\lceil q \rceil$ 的 bin，線性插值恢復分位數

此估計器精確（誤差不超過 bin 寬度）、便宜（僅需一次整數 All-Reduce，代價不到原始 margin 通信的 1%），且估計正確量（計數可加，全球直方圖精確等於 Pooled 全域分佈，與 Token 分片方式無關）。

### 附錄 E：MoonEP 一般上界證明

對給定路由器輸出 $I$，令 $M(I) = \min_P \max_r \{ m_r(P) \}$ 為最優執行計劃下每個 EP rank 的最大冗餘專家數。定理 1：$M(I) \leq E/R$（$E$ 為每設備專家數，$R$ 為 EP ranks 數）。構造證明：反覆從超載 rank 向欠載 rank 遷移 Token 直到完美均衡，每次填補使一個欠載 rank 達到均衡且此後不再改變，過程在 $R-1$ 次內終止，且每個 rank 的遠程 Token 來自單一其他 rank，因此冗餘不超過 $E/R$。

### 附錄 F：Chat Template（XTML）

Kimi K3 的 Chat Template 重新設計為圍繞三個目標：
1. **可擴展性**：新能力通過向後兼容的消息格式引入，無需模板修訂
2. **低對齊稅**：格式以最少監督數據學習，支持輕微微調後直接進入 RL 的流程
3. **解碼友好**：結構容許簡單編碼器、流式解析器和語法約束執行器

Template 採用 XTML（eXtensible Token Markup Language），用三個保留的特殊 Token 替代 XML 語法：`[open]`、`[sep]` 和 `[close]`，加上一個生成停止標記 `[end_of_msg]`。

消息結構分為兩個類別：**輸入消息**（序列化請求的 messages 字段）和**選項消息**（將請求選項轉換為模型在上下文中讀取的指令）。全局選項（工具聲明、推理努力設置）在所有輸入消息之前；一次性選項（tool_choice、response_format）在輸入消息之後，使每請求更改不影響歷史 KV Cache。

助手消息的 body 按**通道**組織：`think` 攜帶推理軌跡，`response` 攜帶用戶可見答案，`tools` 攜帶工具調用。兩種生成模式通過生成前綴選擇：`[open]think[sep]` 為思考模式，`[open]response[sep]` 為指令模式。Kimi K3 只支持保留式思考（preserved thinking）：在思考模式下，`think` 通道始終保留在歷史中，即便其內容為空——這樣模型在多輪對話中觀察到的消息結構保持一致；在指令模式下，助手消息僅包含 `response` 和 `tools` 通道。

**工具調用**：`tools` 通道中每個工具調用攜帶 `tool` 和 `index` 屬性；`index` 與消息內各並行調用的順序一致，每個工具結果消息重複相同的 `tool/index` 配對並按工具結果區塊的順序排列，確保結果與調用無歧義關聯。參數帶類型：字串參數以純文本呈現，其他 JSON 類型的值以緊湊序列化格式。代碼等自由文本因此是一等公民而非轉義的 JSON 字串。純 JSON 回退區塊用於無法解碼為帶類型參數區塊的輸入；它僅出現在輸入 Token 中，不會出現在模型輸出中，且其損失被遮罩。

**推理努力與選項**：Chat Template 支持動態加載工具：對話中檢索或加載的工具透過 `tool-declare` 消息宣告，此後模型的可用工具集擴展而無需重建歷史 KV Cache。

---

## 譯註（Translator's Notes）

> 以下概念說明基於 research 整理，幫助讀者理解技術背景。

**¹ Delta Rule（Delta Rule 遞歸）**

這裡的 Delta Rule 不是神經網絡中用於梯度下降的 delta rule。它來自於「線性注意力是某種快速權重記憶系統」的思想（Mackay, 1990; Schmidhuber, 1992）。本質上：狀態 $S_t$ 儲存「如何根據新的 Query 權重重新加權歷史 Values」，Delta Rule 的含義是「相對於前一個狀態的增量修改」，而不是「覆蓋」。通道級衰減 $\alpha_t$ 控制歷史信息保留多少，$\beta_t$ 控制新信息寫入強度。

**² Linear Attention vs Softmax Attention**

標準 softmax attention 需要對所有先前 token 計算 attention score，複雜度 $O(n^2)$。Linear attention 將其重寫為遞歸形式，複雜度降到 $O(n)$。代價是：linear attention 是全局的（每個 token 可見所有歷史），但缺乏 softmax 的「選擇性」——它不能根據內容動態決定關注什麼。KDA 透過 gated delta mechanism 部分解決這個問題：$\beta_t$ 閘控允許模型根據輸入內容決定「寫入多少新信息」。

**³ MoE Load Balancing / Dead Experts**

在 MoE 中，如果某些專家被選中的頻率遠高於其他專家，會造成：① 這些專家所在設備通信成為瓶頸；② 未被選中的專家（「dead experts」）長期缺乏訓練，權重退化。傳統輔助損失方法對不平衡加懲罰，但這直接干擾路由決策——損失函數的兩個目標（預測準確 + 負載均衡）相互矛盾，訓練動態複雜。輔助損失-free 方法（BASE Layers, Expert Choice, QB）則只修改路由偏置項，不改變 router 的梯度優化方向。

**⁴ NoPE (No Position Encoding)**

標準 Transformer 需要某種方式告訴模型 token 的位置（因為注意力是位置無關的）。RoPE 通過旋轉矩陣將位置編碼融入 query/key；ALiBi 通過 distance-based penalty 實現。NoPE 的思想是：如果注意力機制本身具有「順序敏感性」（例如遞歸狀態攜帶了位置信息），那就不需要顯式位置編碼。KDA 的遞歸狀態本身就攜帶了「哪個 token 是哪個」的相對位置信息，因此不需要 NoPE。

**⁵ Tensor Core vs CUDA Core**

GPU 上執行矩陣乘法有兩種方式：CUDA Core 是通用計算單元，可以執行任意精度、任意形狀的操作；Tensor Core 是專門為矩陣乘法設計的硬體單元，在 Volta 架構（2017）引入後顯著加速了深度學習中的矩陣運算。Tensor Core 要求輸入矩陣滿足特定形狀（通常是 16×16 或 8×8 的小塊），因此稱為「密集矩陣乘法」。KDA 的下界有界衰減確保所有 tile 都滿足這個要求，使整個計算都落在 Tensor Core 上，大幅提升效率。

**⁶ CUTLASS / Triton**

CUTLASS 是 NVIDIA 開發的 CUDA C++ 模板庫，提供高效矩陣乘法和其他底層操作的參考實現，是開發自定義 GPU kernel 的基礎。Triton 是 OpenAI 開發的 DSL（領域特定語言），允許用 Python-like 語法編寫高效 GPU kernel，編譯器自動處理記憶體合併、共享內存分配等問題。相比 PyTorch 的 autograd，寫自定義 kernel 可以：
- 融合多個操作減少記憶體傳輸
- 針對特定硬體優化記憶體訪問模式
- 實現 PyTorch 不支持的運算（如 KDA 的遞歸狀態傳遞）

**⁷ On-Policy Distillation**

On-Policy 意味着用當前策略生成樣本再用這些樣本更新策略。Multi-Teacher On-Policy Distillation（MOPD）是說：有多個不同領域/推理努力級別的教師模型，各自擅長不同的任務；用這些教師生成的樣本蒸餾到一個統一模型中，使統一模型同時具備多個領域的能力。

**⁸ Muon 優化器**

Muon，全稱 **MomentUm Orthogonalized by Newton-Schulz**，由 Keller Jordan 在 2024 年提出。核心思想：

1. 先用 SGD-Momentum 生成梯度更新 $G$
2. 對 $G$ 應用 **Newton-Schulz 迭代**（矩陣正交化後處理步驟），使更新矩陣趨近正交
3. 正交化後的梯度應用於參數

為什麼要正交化？因為非正交梯度更新會導致梯度方向之間的相互干涉，類似於病態條件數問題。Newton-Schulz 迭代是一種收斂快速的矩陣正交化方法（5 步迭代），使更新方向相互正交，改善收斂穩定性。

**Per-Head Muon** 將正交化應用於每個注意力頭的分塊矩陣，而非整個 Q/K/V 投影矩陣。原始 Muon 對整個矩陣做正交化，大梯度/大動量的頭會主導更新方向，小尺度的頭得到不足的歸一化更新；Per-Head 版本讓每個頭獨立正交化，學習動態更均衡。

**⁹ RMSNorm**

Root Mean Square Layer Normalization。與標準 LayerNorm 不同，RMSNorm 只計算 RMS（均方根）而不用均值，在某些場景下訓練更穩定且計算更簡單。

---

*本翻譯基於 Kimi K3 Technical Report (2026-07)，原文版權歸 Kimi Team 所有。翻譯僅供學習參考。*