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

![Kimi K3 整體架構 Figure 2](/higumalu-note/images/k3_tech_report/page03.png)
*圖 2：Kimi K3 架構，按 Token、Channel 和 Layer 三個混合維度組織，輸入端有原生視覺通道。每個 Block 包含三層 KDA 與一層 Gated MLA，每層注意力配對一個 Stable LatentMoE 前饋網絡。AttnRes 使用學習的偽查詢（w）來計算對 Embedding 和先前 Block 輸出的注意力權重（α），實現跨深度的選擇性信息流。左上：Stable LatentMoE 模塊（共享專家 + 路由專家）；左下：KDA 模塊；右下：原生視覺通道*

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

![KDA 下限有界衰減 Figure 3](/higumalu-note/images/k3_tech_report/page05.png)
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

$$s = \text{Softmax}(\text{Norm}(W_r x)) \in \mathbb{R}^n$$

其中 $n = 896$ 是路由專家數，Norm 是可學習的通道級 Normalization。為提高路由穩定性，門控分支計算：

$$\text{gate} = \text{Sigmoid}(W_g x) \in (0, 1)$$

最終權重為 $w = \text{gate} \odot s$。為處理路由中的不平衡，我們引入每專家偏置 $b \in \mathbb{R}^n$，並定義 Top-k 選擇和權重：

$$\mathcal{T}_i = \text{argtop}_k(s_i + b), \quad w_{i,j} = \begin{cases} \frac{\text{Softmax}(s_{i,j} + b_j)}{\sum_{j' \in \mathcal{T}_i} \text{Softmax}(s_{i,j'} + b_{j'})} & j \in \mathcal{T}_i \\ 0 & \text{otherwise} \end{cases}$$

偏置 $b$ 從路由概率計算中省略，這樣它只調節 dispatch 而不改變混合權重或直接干擾梯度優化。

#### 2.3.1 SiTU-GLU

在 FFN 投影之間，我們用 SiTU-GLU 取代標准 SwiGLU。SiTU-GLU 將 Swish 激活函數的 cap 替換為平滑的 Tanh cap：

$$\text{SiTU-GLU}(x) = \beta_1 \tanh\left(\frac{W_g x}{\beta_1}\right) \odot \beta_2 \tanh\left(\frac{W_u x}{\beta_2}\right)$$

Gate branch 的 $\beta_1 = 4$，Up branch 的 $\beta_2 = 25$。由於 $|\tanh(z)| < 1$ 且 $0 < \text{Sigmoid}(z) < 1$，每個輸出坐標滿足 $\|\text{SiTU-GLU}(x)\|_\infty \leq \beta_1 \beta_2 = 100$，輸出有嚴格上界。在原點附近，$\beta \tanh(z/\beta) = z + O(z^3 / \beta^2)$，因此 SiTU-GLU 與 SwiGLU 一階匹配。隨 $\beta \to \infty$ 精確退化為 SwiGLU。

#### 2.3.2 Quantile Balancing

隨著專家數增長到近千，均衡路由變得越來越重要。傳統輔助損失方法 [30] 在每個 Token 上增加一個負載不平衡罰項，但這會直接干擾路由決策與梯度優化。我們採用無輔助損失的路由均衡：對每個專家維護一個偏置項 $b_j$，根據目標加載 $q := mk/n$ 週期性更新：

$$b_j \leftarrow b_j - \eta \left( \frac{1}{m} \sum_i \mathbf{1}_{j \in \mathcal{T}_i} - \frac{k}{n} \right)$$

此更新將專家加載推向目標，類似於 SignSGD，但直接作用於偏置而非路由器參數。

![Quantile Balancing Figure 5](/higumalu-note/images/k3_tech_report/page08.png)
*圖 5：Quantile Balancing 示意（m=8 tokens, n=4 experts, k=1）。(a) 不均衡路由：初始加載 (4, 3, 1, 0)；(b) 分位數平衡：每列繪製偏差後的分數灰條，紅色虛線為偏差調整線；(c) 均衡路由：最終加載 (2, 2, 2, 2)*

對於精確的偏差估計，我們從最優平衡分配問題的對偶推导出 Alternating Quantile Solver：交替求解 token 側閾值 $\alpha$ 和專家側閾值 $\beta$，每步封閉形式精確求解。實務上，用 1000 個 bin 的直方圖估計分位數，僅需一次整數 All-Reduce 通信，代價不到原始 margin 通信的 1%。

---

### 2.4 原生視覺

Kimi K3 的視覺編碼器 MoonViT-V2 將圖像和視頻編碼為視覺 Token 序列。輕量級投影器將視覺特徵映射到與語言骨幹共享的 Embedding 空間，然後一起送入骨幹網絡處理。這種原生多模態設計使視覺理解與語言推理深度融合，而非事後拼接。

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

![擴展法則 Figure 7](/higumalu-note/images/k3_tech_report/page11.png)
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

---

## 4. 後訓練（Post-Training）

### 4.1 強化學習

Kimi K3 的後訓練明確為 100 萬上下文測試時 scaling 設計。跨四個領域開展強化學習，每個領域覆蓋多個推理努力層級（low / medium / high / max）。

**通用領域**：可驗證搜索、專業知識工作，訓練模型在長上下文中的精確信息檢索與綜合能力。

**Agentic 領域**：持久化助理工作流、網頁開發、自主執行任務，訓練數百至數千次工具調用、數百萬累計上下文 Token 的長程執行能力。

**Coding 領域**：軟體工程、內核優化，訓練模型在複雜代碼庫中的導航、修改和調試能力。

**多模態領域**：視覺 in-the-loop 工具使用，訓練模型結合視覺理解和工具調用的能力。

**領域專門化策略的統一**：每個領域和努力層級的專門化策略透過多教師 On-Policy 蒸餾 [76, 135, 29] 整合到統一模型中。

### 4.2 Long-Context SFT

在強化學習之前，Kimi K3 在長上下文任務上接受監督微調（SFT），涵蓋長程推理、多文檔問答和代碼補全。關鍵是使用真實的長上下文數據而非簡單的序列拼接。

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

### 5.3 百萬 Token Agentic 強化學習

百萬 Token 軌跡的強化學習帶來全新的系統挑戰：Rollout 和環境狀態可能跨越數百萬 Token，必須持久化以支持長程執行。

**協作 RL 系統**：我們的協作系統結合部分 Rollout（長軌跡分段執行）、外部 KV-Cache 保留（跨分段持久化狀態）、自適應節流（根據環境響應時間動態調整執行節奏）和可恢復 MicroVM Sandbox（每個 Rollout 在隔離的微型 VM 中執行，狀態可暫停/恢復）。

---

## 6. 評估（Evaluation）

### 6.1 主要結果

![Benchmark 結果 Table 2](/higumalu-note/images/k3_tech_report/page27.png)
*表 2：跨推理、Coding、Agentic、Vision 四大領域的基準測試對比*

**知識/推理領域**

| Benchmark | Kimi K3 | Claude Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2 |
|-----------|---------|----------------|-------------|----------|---------|---------|
| GPQA Diamond | 93.5 | — | — | — | — | — |
| CritPt | **74.7** | 71.2 | 71.8 | — | 65.3 | — |
| AA-LCR | **59.5** | — | — | — | — | — |
| HLE-Full | 43.5 / 56.0 | — | — | — | — | — |

**Coding 領域**

| Benchmark | Kimi K3 | Claude Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2 |
|-----------|---------|----------------|-------------|----------|---------|---------|
| DeepSWE | 67.5 | 72.1 | **73.0** | 46.2 | 59.0 | 64.2 |
| ProgramBench | **77.8** | 71.7 | 72.9 | 69.0 | 64.8 | 64.2 |
| Terminal-Bench 2.1 | 67.0 | 67.5 | **67.5** | 64.8 | 61.8 | 59.6 |
| Fronti

---

**Coding 領域（續）**

| Benchmark | Kimi K3 | Claude Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2 |
|-----------|---------|----------------|-------------|----------|---------|---------|
| FrontierSWE | 84.6 | **88.0** | 71.3 | 63.7 | 64.9 | 66.7 |
| SWE-Marathon | **42.0** | 14.0 | 39.0 | 13.0 | 35.0 | 14.0 |
| PostTrainBench | 83.4 | **88.8** | — | — | — | — |

**Agentic 領域**

| Benchmark | Kimi K3 | Claude Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2 |
|-----------|---------|----------------|-------------|----------|---------|---------|
| BrowseComp | **93.5** | 89.9 | 91.3 | 70.8 | 88.0 | 71.9 |
| DeepSearchQA (F1) | **95.0** | 88.3 | — | — | — | — |
| JobBench | 54.3 | **57.4** | 45.4 | 48.4 | 43.4 | 38.3 |
| AutomationBench | **30.8** | 29.1 | 29.7 | 27.2 | 22.7 | 12.9 |

**Vision 領域**

| Benchmark | Kimi K3 | Claude Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2 |
|-----------|---------|----------------|-------------|----------|---------|---------|
| OmniDocBench | **91.1** | 86.8 | — | — | — | — |
| MMMU-Pro | 71.9 / 83.7 | — | **75.2 / 84.4** | — | — | — |
| CharXiv (RQ) | 88.0 / **90.4** | **91.2** | 84.4 | 82.7 | 84.3 | — |

---

## 7. 案例研究（Case Studies）

![GPU 內核優化案例 Figure 14](/higumalu-note/images/k3_tech_report/page33.png)
*圖 14：AttnRes 內核優化案例，Kimi K3 +59.7% 與 Claude Fable 5 +57.1% 並列第一*

### 7.1 GPU 內核優化

Kimi K3 在一個標準化配置的沙箱中優化 GPU 內核（每任務最多 24 小時），評估四個代表性內核：AttnRes、DeepSeek Sparse Attention（DSA）、KDA 和 MLA（head dimension 512）。這些在 NVIDIA Hopper GPU 和替代供應商 GPGPU 上運行。

**結果**：Kimi K3 將 AttnRes 延遲從 283.6ms 降至 114.4ms（**提升 59.7%**），與 Claude Fable 5（+57.1%）並列第一。DSA 加速 55.1%，KDA 加速 73.6%，MLA 超過 50% 峰值 TFLOPS。

### 7.2 MiniTriton 編譯器開發

Kimi K3 從零開發了一個完整的 Triton-like 編譯器 MiniTriton，包含自定義 tile 級 Python 前端、Warp 級 MLIR 標注/優化層和 PTX 代碼生成管線。在 NVIDIA L20 上，MiniTriton 在幾何平均意義下超越 PyTorch eager 和 `torch.compile`。

### 7.3 晶片設計

早期概念驗證：Kimi K3 為 nano 模型設計推理晶片原型，採用混合 KDA + NoPE-MLA 注意力、Block AttnRes（block size=2）、Group-wise INT4 量化權重。在 4 mm² 面積預算內達到 100 MHz 時序閉合，RTL 模擬吞吐量 > 8,700 tokens/s。

---

## 8. 相關工作（Related Work）

本節回顧與 Kimi K3 設計相關的先前工作，涵蓋注意力機制、MoE 架構、測試時 scaling 和基礎設施優化等領域。

**注意力機制**：標準 Transformer 的 softmax 注意力在序列長度上具有 $O(n^2)$ 複雜度。線性注意力 [142, 141] 將其降低到 $O(n)$，代價是引入局部性偏差。Mamba [49] 和 Gated Delta Networks [139] 引入選擇性狀態空間機制。KDA 在此基礎上增加了通道級遺忘門和下限有界衰減，實現硬體高效實現。

**專家混合**：MoE [33, 23] 通過稀疏激活減少計算成本。Switch Transformer [33] 使用簡單的 Top-1 路由，DeepSeekMoE [23] 引入了細粒度專家分割和共享專家。BASE Layers [68] 和 Expert Choice [117] 從最優傳輸角度重新定義路由。Kimi K3 的 Quantile Balancing 從最優平衡分配的對偶推导出，數學嚴謹且收斂快速。

**測試時 Scaling**：OpenAI o-series [85, 84]、Anthropic Extended Thinking [6, 7]、DeepSeek-R1 [41] 和 Kimi K1.5 [119] 展示了 RL 驅動推理能力提升的巨大潛力。Kimi K3 在此基礎上將測試時 scaling 擴展到百萬 Token 長度和多模態場景。

---

## 附錄（Appendices）

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

助手消息的 body 按**通道**組織：`think` 攜帶推理軌跡，`response` 攜帶用戶可見答案，`tools` 攜帶工具調用。兩種生成模式通過生成前綴選擇：`[open]think[sep]` 為思考模式，`[open]response[sep]` 為指令模式。

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

Muon = "Matrix u for u, Newton-ish"，是一種牛頓類優化器，專門為 Transformer 中的矩陣參數設計。它使用 Newton-Schulz 正交化來維持權重矩陣的正交性，聲稱在某些設置下收斂速度比 AdamW 快。Per-Head 版本將正交化應用於每個注意力頭的分塊矩陣，而非整個 Q/K/V 投影矩陣，避免不同 head 因梯度尺度不同而在更新中相互干擾。

**⁹ RMSNorm**

Root Mean Square Layer Normalization。與標準 LayerNorm 不同，RMSNorm 只計算 RMS（均方根）而不用均值，在某些場景下訓練更穩定且計算更簡單。

---

*本翻譯基於 Kimi K3 Technical Report (2026-07)，原文版權歸 Kimi Team 所有。翻譯僅供學習參考。*