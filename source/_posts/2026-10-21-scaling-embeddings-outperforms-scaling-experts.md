---
title: 美團 LongCat 團隊技術報告翻譯：擴展嵌入層在語言模型中超越擴展專家（Scaling Embeddings Outperforms Scaling Experts in Language Models）
date: 2026-10-21 10:30:00
updated: 2026-10-21 10:30:00
categories: [AI, 論文翻譯]
tags: [美團, LongCat, MoE, N-gram Embedding, 稀疏架構, 投機解碼, 論文翻譯]
description: 美團 LongCat 團隊最新技術報告深入探索稀疏參數擴展範式：證明擴展 N-gram Embedding 參數在特定區間相比擴展 MoE 專家能取得更優的 Pareto 前沿，並開源 68.5B 參數（僅激活約 3B，嵌入佔 31.4B）的 LongCat-Flash-Lite，全面超越同等參數 MoE 基線。本文提供全量逐字精準翻譯與深入架構技術譯註。
mathjax: true
---

> **論文標題**：Scaling Embeddings Outperforms Scaling Experts in Language Models  
> **作者**：Hong Liu, Jiaqi Zhang*, Chao Wang, Xing Hu, Linkun Lyu, Jiaqi Sun, Xurui Yang, Bo Wang, Fengcun Li, Yulei Qian, Lingtong Si, Yerui Sun, Rumei Li, Peng Pei*, Yuchen Xie, Xunliang Cai (*通訊作者: zhangjiaqi39@meituan.com, peipeng@meituan.com)  
> **機構**：美團 LongCat 團隊 (Meituan LongCat Team)  
> **發布日期**：2026 年 1 月 26 日（v2 更新：2026 年 2 月 11 日）  
> **論文鏈接**：[arXiv:2601.21204](https://arxiv.org/abs/2601.21204)  
> **開源權重**：[Hugging Face: meituan-longcat/LongCat-Flash-Lite](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite)  

<!-- more -->

---

## 摘要（Abstract）

儘管專家混合（Mixture-of-Experts, MoE）架構已成為大型語言模型（LLM）稀疏擴展的標準範式，但它們正面臨日益嚴重的邊際效益遞減與系統級瓶頸。在本研究中，我們探索了**嵌入層擴展（Embedding Scaling）**作為擴展稀疏性的一條強大且正交的全新維度。

透過全面的分析與實驗，我們指出了**嵌入擴展相比專家擴展能夠取得更優 Pareto 前沿（Superior Pareto Frontier）的特定運作區間**。我們系統性地刻畫了決定此效能的關鍵架構要素——涵蓋從**參數預算分配（Parameter Budgeting）**到與**模型寬度及深度**的交互作用。此外，透過結合針對性的系統優化與**投機解碼（Speculative Decoding）**，我們成功將這種參數稀疏性轉化為實質的推理加速。

在這些見解的指導下，我們推出並開源了 **LongCat-Flash-Lite**——一個總參數達 68.5B、從頭訓練（trained from scratch）、每個 Token 激活僅約 3B 的模型。儘管將超過 30B 的參數分配給了嵌入層，LongCat-Flash-Lite 不僅全面超越了參數量相當的 MoE 基線，而且在可比規模的現有模型中展現出卓越的競爭力，尤其在 **Agent 工具調用**與 **Coding 程式碼任務**領域表現尤為突出。

---

## 1. 引言（Introduction）

專家混合（Mixture-of-Experts, MoE）架構已確立為擴展大型語言模型（LLM）的主導範式（Lepikhin et al., 2021），它使模型能夠在維持可控計算成本的同時擴展至龐大的參數量。透過動態地將 Token 路由至專家的子集，模型將參數容量與計算成本解耦，允許 LLM 擴展至數兆參數，同時維持適度的推理延遲。然而，隨著模型規模和稀疏度水平的提升，性能的邊際收益逐漸減弱，最終趨向效率飽和點（Abnar et al., 2025）。此外，專家的實際擴展受到系統級瓶頸的約束，特別是分散式訓練中日益升高的通訊開銷與記憶體頻寬壓力。這使得探索前饋網絡（FFN）之外擴展稀疏參數的替代性正交維度變得勢在必行。

與 MoE 相比，嵌入層提供了一個**被忽視、本質具備稀疏性、且查表複雜度僅為 $\mathcal{O}(1)$ 的維度**。這允許海量的參數擴展而**無需路由開銷**——實際上在不引發計算爆炸的前提下實現了參數擴充。該維度的理論基礎由詞表擴展定律（Scaling laws with vocabulary; Tao et al., 2024）建立，該定律指出較大的模型需要成比例擴大的詞表以最大化計算效率。為了發掘這一潛力，學界與業界提出了多種策略：
1. **結構擴展（Structural Expansion）**：以逐層嵌入（Per-Layer Embedding, PLE; Google DeepMind, 2025; Sadhukhan et al., 2026; bcml-labs, 2025）為代表，它為每一層分配獨立的嵌入參數以擴展容量。
2. **詞表擴展（Vocabulary Expansion via N-grams）**：透過 N-gram 增加每個 Token 攜帶的信息密度。這一概念可追溯至 RNN 時代的查表語言模型（Lookup-table language models; Huang et al., 2021），並近期在 LLM 中得到推進（Clark et al., 2022; Huang et al., 2025; Pagnoni et al., 2025; Cheng et al., 2026）。

這些方法共同突顯出嵌入層是擴展模型能力的肥沃土壤。

儘管近期對擴展 LLM 嵌入參數的興趣濃厚，但仍有數個關鍵挑戰尚未得到充分探索：
* 首先，專家參數與嵌入參數之間的**相對擴展效率**尚不明確，使得這兩個稀疏維度之間的最優容量分配比例模糊不清；
* 其次，擴展嵌入的**約束邊界**尚未得到系統性刻畫：總參數預算、詞表大小、初始化方案以及模型寬度與深度的權衡等因素如何共同影響嵌入擴展的有效性與穩定性，仍缺乏清晰解答；
* 第三，雖然已提出多種擴展嵌入的方法，但在不同區間下哪種策略更有效且高效仍不明朗；
* 最後，擴展嵌入改變了模型在解碼過程中的輸入/輸出特徵，可能影響整體的 I/O 效率，但其對端到端推理性能的影響仍缺乏充分的分析與系統級優化。

在本技術報告中，我們提出一項研究來解決這些挑戰，並建立一個強大的嵌入擴展框架。我們的主要貢獻總結如下：
* **嵌入擴展 vs. 專家擴展之對比**：透過跨多種情境的全面擴展實驗，我們識別出了嵌入擴展相比增加專家數量能取得更優 Pareto 前沿的特定運作區間，為模型擴展提供了一種高效替代方案。
* **架構因素之影響分析**：我們確立了決定嵌入擴展效能的完整架構要素集合，涵蓋整合時機、參數預算分配、雜湊衝突、超參數設定與嵌入初始化，以及模型寬度和深度的影響。此外，我們研究了擴展嵌入的不同方法，發現 N-gram Embedding 展現出最強健的可擴展性。
* **推理效率與系統優化**：我們證明 N-gram Embedding 大幅減輕了 MoE 層中的 I/O 瓶頸，特別是在搭配投機解碼以最大化硬體利用率時。針對隨之而來的嵌入開銷，我們提出了專門的 **N-gram Cache** 與同步內核，確保激活參數的減少直接轉化為更低的延遲與更高的吞吐量。
* **開源 LongCat-Flash-Lite**：基於上述發現，我們推出並開源了 LongCat-Flash-Lite——一個從頭訓練、總參數 68.5B、根據語境動態激活 2.9B~4.5B 參數的模型。評測表明，LongCat-Flash-Lite 不僅超越了參數量相當的 MoE 基線——證實了將超過 30B 參數分配給嵌入而非專家的卓越優勢——而且在同等規模現有模型中展現出高度競爭力，尤其在 Agent 工具調用與 Coding 任務上領先。

---

## 2. N-gram 嵌入層（N-gram Embedding Layer）

為了擴展嵌入參數，我們採納了 Clark et al. (2022); Huang et al. (2025); Pagnoni et al. (2025) 提出的 N-gram Embedding，透過擴展無詞表（vocabulary-free）的 N-gram 嵌入表來增強嵌入模組的表徵能力。

具體而言，對於序列中的第 $i$ 個 Token $t_i$，增強後的嵌入 $e_i$ 計算如下：

$$e_i = \frac{1}{N} \left( E_0(t_i) + \sum_{n=2}^N E_n(\mathcal{H}_n(t_{i-n+1}, \dots, t_i))  \right) 	ag{1}$$

其中當 $j \le 0$ 時 $t_j = 0$。$E_0 \in \mathbb{R}^{V_0 	imes D}$ 是隱維度為 $D$ 的原始基礎嵌入表，$E_n \in \mathbb{R}^{V_n 	imes D}$ 是擴展的嵌入表，$N$ 表示最高 N-gram 階數，$\mathcal{H}_n$ 表示雜湊映射函數。

我們採用**多項式滾動雜湊函數（Polynomial Rolling Hash Function）**：

$$\mathcal{H}_n(t_{i-n+1}, \dots, t_i) = \left( \sum_{j=0}^{n-1} t_{i-j} * V_0^j  \right) \% V_n 	ag{2}$$

為了進一步提升模型的表達能力並減少雜湊衝突，Clark et al. (2022); Huang et al. (2025) 將每個 N-gram 嵌入表分解為 $K$ 個具有不同詞表大小的子表（sub-tables）。Huang et al. (2025) 進一步引入額外的線性投影將輸出映射回原始嵌入空間。N-gram 嵌入的最終形式（在 Huang et al., 2025 中亦被稱為 Over-Encoding）如圖 1 所示，可寫為：

$$e_i = \frac{1}{(N-1)K + 1} \left( E_0(t_i) + \sum_{n=2}^N \sum_{k=1}^K W_{n,k} E_{n,k}(\mathcal{H}_{n,k}(t_{i-n+1}, \dots, t_i))  \right) 	ag{3}$$

其中 $E_{n,k} \in \mathbb{R}^{V_{n,k} 	imes \frac{D}{(N-1)K}}$ 為子表，$W_{n,k} \in \mathbb{R}^{D 	imes \frac{D}{(N-1)K}}$ 為線性投影矩陣。

透過將子表的隱層大小設定為與子表數量成反比，該設計保證了 **N-gram 嵌入的參數量相對於 $N$ 和 $K$ 保持恆定（Invariant）**。

![N-gram 嵌入層架構](/higumalu-note/images/scaling_embeddings/NE-overview.svg)
*圖 1：N-gram 嵌入層架構（Huang et al., 2025）。每個 Token 的嵌入表徵由 N-gram 嵌入分支進行增強。*

---

## 3. 專家擴展與嵌入擴展之對比分析（Comparative Analysis of Expert and Embedding Scaling）

本節介紹我們關於擴展嵌入與擴展專家的實證比較研究結果。

* **實驗設置**：我們將 N-gram 嵌入整合至 Longcat-Flash 架構（Meituan, 2025）中，並在不同激活參數預算（280M、790M 和 1.3B）下進行從頭預訓練的 Scaling 實驗。
* 具體而言，對於 N-gram 嵌入擴展，我們首先訓練基礎稀疏度介於 35% 到 98% 之間的 MoE 模型，並從特定稀疏度水準開始逐步引入 N-gram 嵌入。至關重要的是，在每個稀疏度水準下，N-gram 嵌入模型都與一個**參數量完全對等（Parameter-equivalent）的 MoE 基線**配對——該基線純粹透過增加專家數量達到相同的總參數量。
* 所有模型均在 300B Token 的語料庫上進行預訓練。我們透過監控在精心構建的中英文雙語驗證集上的訓練損失與驗證損失來評估性能。

### 3.1 引入 N-gram 嵌入的最佳時機（Optimal Timing for N-gram Embedding Integration）

一項關鍵發現是：**N-gram 嵌入的擴展動態強烈依賴於基礎模型的稀疏度水準。**

![MoE 與 N-gram Embedding Scaling 曲線對比](/higumalu-note/images/scaling_embeddings/scaling-200M_pol_n.svg)
*圖 2：MoE 模型與 N-gram 嵌入（NE）模型的 Scaling 曲線。橫軸為總參數與激活參數（280M）的比率（作為稀疏度的代理指標）。右側子圖轉換為對數刻度。對於兩條 NE 曲線，虛線連接對應未加入 NE 的基礎 MoE 模型。*

圖 2 展示了三條不同的 Scaling 軌跡：
1. 標準 MoE 基線（藍線）；
2. 應用於低參數比基礎模型的 N-gram 嵌入（綠線）；
3. 應用於高參數比基礎模型的 N-gram 嵌入（紅線）。

圖 2 顯示，MoE 的擴展曲線嚴格遵循對數線性關係。這意味著在低比率區間，邊際增加專家數量可以大幅降低損失；反之，在較高比率區間，要達到相同的損失降低需要顯著更龐大的專家參數增量。

因此，**當在低參數比時引入 N-gram 嵌入，其擴展優勢不及單純增加專家數量帶來的收益；然而，在高稀疏度水準下，N-gram 嵌入的效益變得顯著更加突出。** 這一觀察引出了引入 N-gram 嵌入的第一項設計原則：

> **核心總結 1**：應當在專家數量超過其「甜蜜點（Sweet Spot）」時，再引入 N-gram Embedding。

這表明嵌入擴展是與專家擴展正交的一條極富前景的擴展維度。

---

### 3.2 整合策略（Integration Strategy）

#### 3.2.1 N-gram 嵌入的參數預算分配（Parameter Budgeting for N-gram Embeddings）

進一步觀察圖 2 可以發現藍色曲線與紅色曲線之間存在一個明確的交點：**隨著參數比率持續上升，N-gram 嵌入的性能優勢逐漸收斂，最終被 MoE 基線反超。**

這表明當模型將過高比例的參數預算分配給 N-gram 嵌入時，其性能會劣於同等參數量之 MoE 基線。這與同期工作 Engram（Cheng et al., 2026）的結論不謀而合——損失作為 N-gram 嵌入比例的函數呈現 U 型 Scaling 曲線。在圖 2 中，交叉點略高於 20 的參數比；此時 N-gram 嵌入參數約佔總參數量的 50%（因為基礎 MoE 模型的比例為 12）。由此我們得出第二項原則：

> **核心總結 2**：分配給 N-gram Embedding 的參數預算不應超過總參數量的 50%。

#### 3.2.2 透過詞表定規緩解雜湊衝突（Mitigating Hash Collisions via Vocabulary Sizing）

在 N-gram 嵌入中，雜湊衝突迫使單一嵌入向量疊加多個不同 N-gram 的語義，損害學習效率並導致性能下降。我們發現**選擇合適的詞表大小對抑制高衝突率至關重要**。在訓練中，我們觀察到 N-gram 嵌入在特定詞表大小下會出現異常高的雜湊衝突率，特別是在 2-gram 雜湊中。為探究其底層機制，我們進行了雙重分析：
1. **詞表命中率（Vocabulary Hit Rate）**：預訓練語料庫至少激活一次的詞表條目比例；
2. **雜湊衝突（Hash Collisions）**：量化取模索引導致唯一 Token 表徵損失的次數。

![詞表命中率與衝突趨勢分析](/higumalu-note/images/scaling_embeddings/hit_rate_n.svg)
![2-gram 雜湊衝突分析](/higumalu-note/images/scaling_embeddings/collision_trend_n.svg)
*圖 3：(a) 不同 N-gram 的詞表命中率；(b) 不同詞表大小下的 2-gram 雜湊衝突數量。為清晰起見，在詞表整數倍附近的取樣點更為密集。*

* 圖 3a 顯示，2-gram 雜湊的命中率呈現平緩增長，而高階 N-gram 雜湊迅速收斂至接近 1.0 的命中率。
* 圖 3b 顯示，2-gram 雜湊衝突數量與詞表大小呈現強烈的非線性相關性。一個顯著模式浮現：**當詞表大小接近基礎詞表大小的整數倍時，衝突數量顯著暴增！** 無論 N-gram 詞表大小是否為質數，該現象均持續存在。結合這些觀察，圖 3b 引出了一條關鍵原則：

> **核心總結 3**：N-gram Embedding 的詞表大小應顯著偏離基礎詞表大小的整數倍，以防止雜湊衝突。

#### 3.2.3 超參數敏感度分析（Sensitivity Analysis of Hyperparameters）

我們檢驗了模型性能對 N-gram 階數 $N$ 和子表數量 $K$ 的敏感度：
* **階數 $N$**：增加 $N$ 能捕捉更豐富的上下文語義，但高階 N-gram 在語料中出現頻率極低，導致詞表上的極度稀疏分佈，大幅增加了學習有效嵌入的難度。
* **子表數量 $K$**：決定了應用於每個 N-gram 的獨立雜湊函數數量，能顯著減輕衝突概率，但超過特定閾值後收益遞減。

![N 與 K 超參數消融曲線](/higumalu-note/images/scaling_embeddings/loss-nk_n.svg)
*圖 4：不同 $N$ 與 $K$ 組合下的訓練與驗證損失對比。*

利用 790M 激活參數模型進行消融（圖 4），當 $N=2$ 且 $K=1$ 時模型性能明顯較差；反之，**只要 $N \ge 3$ 且 $K \ge 2$，不同配置之間的性能差異便相當微小**，表明模型在此區間內對超參數選擇具有很強的強健性。經驗上，將 $N$ 設在 3 到 5 之間即可穩定獲得接近最優的表現。

#### 3.2.4 促進有效訓練的嵌入放大策略（Embedding Amplification for Effective Training）

在早期實驗中，我們觀察到嵌入模組的次優初始化會嚴重阻礙 N-gram 嵌入的效能。在 300B Token 預訓練後，我們計算了所有層的模組輸出與其對應殘差分支（identity path）的 $L_2$ 範數，繪製於圖 5。

![層級輸出範數與殘差比率分析](/higumalu-note/images/scaling_embeddings/norm.svg)
*圖 5：模組輸出與對應恆等分支的 $L_2$ 範數層級分析及其比率。每個 shortcut 層包含兩個子層（後綴 0 與 1）。*

圖 5 揭露了一個重大差距：**第一個注意力模組輸出的 $L_2$ 範數比對應的恆等分支（本質上代表嵌入模組的輸出）大出整整一個數量級（約 $10imes$）！** 這意味著在相加時，注意力輸出完全主導了殘差流，有效「淹沒（drowning out）」了嵌入訊號。這種訊號抑制現象隨訓練進行而顯著加劇。

為緩解此問題，我們探索了兩種策略：
1. **縮放因子（Scaling Factor）**：在嵌入輸出中引入縮放因子（通常為 $\sqrt{D}$），以確保其在前向傳播中有足夠貢獻；
2. **歸一化（Normalization）**：在與殘差分支合併前對嵌入輸出應用 LayerNorm。LayerNorm 在訓練初期強制施加單位方差，同樣放大了嵌入的貢獻。

在我們的實驗中，應用**嵌入放大（Embedding Amplification）**技術使得訓練損失和兩個驗證損失一致性地降低了 **0.02**。

---

### 3.3 跨模型寬度與深度的 Scaling 特性（Scaling Properties across Model Width and Depth）

#### 3.3.1 模型寬度增加放大優勢（Enhanced Advantage in Wider Models）

我們在固定模型深度（10 個 shortcut 層）的前提下，在 790M 和 1.3B 激活參數規模上只改變模型寬度（隱層維度與模組維度）進行 Scaling 實驗。

![790M 與 1.3B Scaling 曲線](/higumalu-note/images/scaling_embeddings/scaling-500M_n.svg)
![1.3B Scaling 曲線](/higumalu-note/images/scaling_embeddings/scaling-1B_n.svg)
*圖 6：(a) 790M 激活規模下的 Scaling 曲線；(b) 1.3B 激活規模下的 Scaling 曲線。*

分析揭示了兩個核心趨勢：
1. 當以合適比例引入時，N-gram 嵌入始終取得低於對等 MoE 基線的損失；
2. **極為關鍵的是：隨著模型寬度（激活規模）的增加，N-gram 嵌入曲線與 MoE 曲線的交點系統性地向更高的總參/激活參比率移動！**
   * 在 280M 激活尺寸下，比率超過 30 時 NE 便劣於 MoE；
   * 在 790M 下，NE 僅在英文驗證集上於該比率落後，其他指標均保持領先；
   * **在 1.3B 激活尺寸下，即使比率高達 50，N-gram 嵌入依然保持明確優勢！**

> **核心總結 4**：在層數固定的情況下，增加模型寬度賦予 N-gram Embedding 更大的優勢。

#### 3.3.2 模型深度增加收益遞減（Diminishing Returns in Deeper Models）

對於 Pre-LayerNorm 架構，隨著網絡深度的增加，N-gram 嵌入透過殘差連接的貢獻本質上會逐漸衰減。

我們在 1.3B 激活配置下訓練了 20 層和 40 層的模型，同時在所有深度中嚴格維持 N-gram 嵌入參數佔總參數約 50% 的比例。

![寬度與深度對損失縮減之影響對比](/higumalu-note/images/scaling_embeddings/loss-diff-width_n.svg)
![深度損失差值](/higumalu-note/images/scaling_embeddings/loss-diff-depth_n.svg)
*圖 7：N-gram 嵌入模型相比基線在不同 (a) 模型寬度 和 (b) 模型深度 下的損失降低幅度。*

圖 7b 呈現出鮮明的對比：**當模型深度超過 20 層時，N-gram 嵌入相對基線的優勢經歷了顯著收縮。** 這與增加寬度時優勢持續擴大的趨勢（圖 7a）形成強烈反差。

> **核心總結 5**：增加模型深度會減弱 N-gram Embedding 的相對優勢。

值得注意的是，當前大多數實用語言模型的 shortcut 層數通常在 40 層以下（相當於 80 個常規層）。鑑於寬度的增加能持續放大 NE 優勢，且其在 40 層下仍有良好表現，在常見架構深度內擴展 N-gram 嵌入仍能斬獲巨大的性能收益。

---

## 4. 高效推理（Efficient Inference）

### 4.1 削減 MoE 激活參數（Reduction of MoE Activation Parameters）

N-gram 嵌入機制有效將參數從 MoE 層重新分配到嵌入空間中。這種架構變換在保持模型總參數不變的同時，**減少了 MoE 層內的激活參數量**——這在大 Token 數的記憶體 I/O 密集型解碼場景中極具優勢。此外，嵌入表的增大不會拖累延遲，因為嵌入查表的計算成本僅隨輸入 Token 數量擴展，而與嵌入總參數量無關。

![激活專家數與解碼吞吐量](/higumalu-note/images/scaling_embeddings/flash_comparison_n2.svg)
![解碼吞吐量](/higumalu-note/images/scaling_embeddings/throughput_n_final_f.svg)
*圖 8：(a) 不同 Batch Size 下 LongCat-Flash-Lite 與 Vanilla 基線激活專家數對比；(b) 8×H800-80G 上 LongCat-Flash-Lite 的解碼性能（ISL=4K, OSL=1K）。*

為了充分發揮激活參數減少帶來的效率優勢，關鍵在於透過大 Batch Size 最大化硬體利用率（如圖 8a）。這與**投機解碼（Speculative Decoding）**形成了天然的協同效應：多步投機解碼實質上擴大了「有效 Batch Size」，從而將參數稀疏性的理論優勢直接轉化為有形的推理加速。

### 4.2 優化嵌入查找（Optimized Embedding Lookup）

儘管將參數從專家重配至 N-gram 嵌入降低了 MoE 的 I/O，但相比標準嵌入層，它引入了額外的 I/O、計算與通訊開銷。

為此，我們引入了 **N-gram Cache**——一種受 KV Cache 設計啟發的專用快取機制。我們實現了客製化的 CUDA 算子直接在設備端管理 N-gram ID，實現與複雜推理調度邏輯的低開銷同步。

在投機解碼場景中，草稿模型（Draft Model）層數更少且延遲更低，此時 N-gram 嵌入的相對開銷會放大。為解決此問題，我們提出兩項互補優化：
1. **草稿模型採用常規嵌入層**，繞過計算成本較高的 N-gram 查找；
2. **在 Drafting 階段快取 N-gram 嵌入**，消除後續驗證階段的重複計算。

### 4.3 重新思考 N-gram 嵌入優化：投機解碼的深層角色

除了硬體效率之外，我們認為 N-gram 嵌入架構本質上編碼了豐富的局部語境與 Token 共現資訊，與投機解碼存在未被發掘的協同潛力：
* **基於 N-gram Embedding 的 Drafting**：由於 N-gram 嵌入聚合了前 $N-1$ 個 Token 的信息，它隱式捕捉了短程依賴性。我們正在探索直接在 N-gram 嵌入輸出上附加輕量線性投影，將其改造為超輕量的草稿模型。
* **Early Rejection（提前拒絕）**：N-gram 嵌入表徵可作為外部草稿模型生成 Token 的語義一致性檢查器（置信度估計器）。若草稿 Token 在 N-gram 空間的匹配機率極低，可在進入昂貴的目標模型驗證前「提前拒絕」，進一步降低驗證工作量。

---

## 5. 與逐層嵌入（Per-Layer Embedding）的整合探討

本節對比 N-gram Embedding 與 Google DeepMind 提出的逐層嵌入（PLE），並介紹兩者的整合嘗試。

### 5.1 逐層嵌入（PLE）
PLE 直接將 SwiGLU 模組中 Up-projection 矩陣的輸出替換為嵌入輸出。設 $x_i^{(l)}$ 為第 $l$ 層 FFN 模組的第 $i$ 個輸入向量，加入 PLE 的 FFN 輸出為：

$$	ext{FFN}^{(l)}(x_i) = W_d^{(l)} \left( 	ext{SiLU}(W_g^{(l)} x_i^{(l)}) \odot E_0^{(l)}(t_i)  \right) 	ag{4}$$

### 5.2 逐層 N-gram 嵌入（PLNE）
我們提出 **Per-Layer N-gram Embedding (PLNE)**，將每層的基礎嵌入輸出替換為 N-gram 嵌入輸出：

$$	ext{FFN}^{(l)}(x_i) = W_d^{(l)} \left( 	ext{SiLU}(W_g^{(l)} x_i^{(l)}) \odot e_i^{(l)}  \right) 	ag{5}$$

其中 $e_i^{(l)}$ 依據公式 (3) 計算，使用各層專屬的嵌入表與投影矩陣。

### 5.3 實證對比

![NE、PLE 與 PLNE 損失對比](/higumalu-note/images/scaling_embeddings/loss-ple_n.svg)
*圖 9：790M 激活參數設定下 N-gram 嵌入（NE）、PLE 與 PLNE 的損失對比。*

圖 9 顯示，PLE 的表現劣於 N-gram Embedding，而 PLNE 相對 NE 僅取得微幅改善。我們將前者歸因於 N-gram 嵌入相比標準嵌入具備更優越的學習效率。

然而，在後續增加模型寬度或深度的實驗中，PLNE 未能展現出持續的優勢，在多數場景下僅與 NE 持平。考慮到 PLNE 由於每層增加大量投影矩陣而實質增加了激活參數量，**我們最終在更大規模的實驗中未採用 PLNE**。

---

## 6. LongCat-Flash-Lite

結合先前的研究見解，美團推出了整合 N-gram 嵌入並從頭訓練的 **LongCat-Flash-Lite**。

### 6.1 模型配置
* **架構細節**：共 14 個 shortcut 層，總參數 68.5B，由於 Zero-Experts 機制，每 Token 動態激活 2.9B 至 4.5B 參數。MoE 模組包含 256 個 FFN 專家與 128 個 Zero-Experts，每 Token 選擇 12 個專家。**嵌入模組包含 31.4B 的 N-gram 嵌入參數，佔總參數量的 46%**。
* **訓練數據**：在 11T Token（8k 長度）上進行預訓練，隨後進行 1.5T Token（擴展至 128k）的 Mid-training，最後進行 SFT。支援高達 256k 的上下文長度。
* **對照基線**：訓練了一個總參數完全相同的 MoE 基線（**LongCat-Flash-Lite-Vanilla**），將 31.4B 的 N-gram 嵌入參數全部轉化為額外的 MoE 專家，兩者遵循完全相同的數據配方與訓練策略。

![訓練損失曲線對比](/higumalu-note/images/scaling_embeddings/lm_loss_n.svg)
*圖 10：LongCat-Flash-Lite 與 Vanilla 基線的平滑訓練損失曲線。*

---

### 6.2 Base 基礎模型評測結果

#### 表 1：LongCat-Flash-Lite 與 Vanilla 基線在 1.3T 訓練節點下的 Base 模型性能對比

| 評測領域 | 基準測試 (Benchmark) | LongCat-Flash-Lite-Vanilla @1.3T | LongCat-Flash-Lite @1.3T |
| :--- | :--- | :---: | :---: |
| **通用領域 (General)** | MMLU | **64.81** | 64.01 |
| | MMLU-Pro | 34.43 | **35.89** |
| | C-Eval | 64.09 | **67.21** |
| | CMMLU | 67.08 | **69.55** |
| **推理領域 (Reasoning)** | BBH | 38.54 | **43.67** |
| | GPQA | 25.37 | **29.66** |
| | DROP | 47.92 | **52.43** |
| | GSM8K | 50.00 | **50.50** |
| **程式碼領域 (Coding)** | HumanEval+ | 28.66 | **31.10** |
| | MultiPL-E | **30.20** | 30.03 |
| | BigCodeBench | 33.42 | **36.05** |

在絕大多數基準測試中，LongCat-Flash-Lite 均大幅領先 Vanilla 基線，特別是在 **BBH (+5.13)**、**GPQA (+4.29)**、**DROP (+4.51)**、**C-Eval (+3.12)** 與 **BigCodeBench (+2.63)** 上優勢尤為突出。

---

### 6.3 Chat 對齊模型橫向對比

#### 表 2：LongCat-Flash-Lite 與同規模前沿模型（Kimi-Linear、Qwen3-Next、Gemini 2.5 Flash-Lite）對比

| 評測維度 | 評測基準 (Benchmark) | Kimi-Linear-48B-A3B | Qwen3-Next-80B-A3B-Instruct | Gemini 2.5 Flash-Lite | LongCat-Flash-Lite |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **基礎架構** | Architecture | MoE | MoE | - | **MoE + NE** |
| **總參數量** | Total Params | 48B | 80B | - | **68.5B** |
| **激活參數量** | Activated Params | 3B | 3B | - | **2.9B ~ 4.5B** |
| **上下文長度** | Context Length | 128k | 128k | 1M | **256k** |
| **Agent 工具調用** | Tau2-Airline (avg@8) | 44.00 | 45.5* | 35.00 | **58.00** |
| | Tau2-Retail (avg@8) | 18.86 | 57.3* | 37.50 | **73.10** |
| | Tau2-Telecom (avg@8) | 15.68 | 13.2* | 21.93 | **72.80** |
| | VitaBench (avg@4) | - | 5.80 | 4.50 | **7.00** |
| **Agent Coding** | SWE-Bench (acc) | 32.80 | 37.60 | 41.3* | **54.40** |
| | TerminalBench (acc) | 20.00 | 15.19 | 20.00 | **33.75** |
| | SWE-Bench Multilingual | 37.20 | 31.30 | - | **38.10** |
| | PRDBench | - | 15.36 | - | **39.63** |
| **通用領域** | GPQA-Diamond (avg@16) | 69.89 | **74.33** | 70.20* | 66.78 |
| | MMLU (acc) | 79.91 | **89.28** | 84.68 | 85.52 |
| | MMLU-Pro (acc) | 67.22 | **82.93** | 78.95 | 78.29 |
| | CEval (acc) | 78.48 | **90.91** | 75.16 | 86.55 |
| | CMMLU (acc) | 76.26 | **86.50** | 72.06 | 82.48 |
| **數學推理** | MATH500 (acc) | 94.20 | **98.00** | 95.20 | 96.80 |
| | AIME24 (avg@32) | 70.52 | **81.35** | 63.33 | 72.19 |
| | AIME25 (avg@32) | 59.58 | **68.44** | 50.1* | 63.23 |

在 **Agentic Tool Use** 與 **Agentic Coding** 任務上，LongCat-Flash-Lite 建立了壓倒性的領先優勢：
* 在 Tau2-Bench 零售場景中達到 **73.10**，電信場景更斬獲驚人的 **72.80**（比其他模型高出數倍）；
* 在真實世界軟體工程基準 SWE-Bench 上達到 **54.40%**，超越 Gemini 2.5 Flash-Lite (41.3) 與 Qwen3-Next-80B (37.60)；
* 在終端指令執行 TerminalBench 上取得 **33.75%**，展現極強的 Agentic 系統操作能力。

### 6.4 內核優化與推理實測加速

為了克服極度稀疏激活帶來的硬體頻寬飽和問題，團隊採用 Eagle3 與 3-step 投機解碼策略，配合寬專家並行（Wide EP）與 SBO（Single Batch Overlap）。為解決隨之而來的內核發射開銷，團隊實施了全方位的算子融合（Communication-Fusion、RMSNorm、SwiGLU 激活量化融合）與 PDL（Programmatic Dependent Launch）技術。

#### 表 3：投機解碼推理加速比

| 硬體環境 | 投機解碼加速比 (Tokens/s) |
| :--- | :---: |
| **8× H800** | **1.81×** |

在 8× H800 GPU 叢集上，端到端投機解碼吞吐實現了 **1.81 倍**的純硬體加速。

---

## 7. 結論（Conclusions）

在本技術報告中，我們挑戰了僅靠增加 MoE 專家來擴展模型容量的傳統思維。我們證明了：**擴展 N-gram 嵌入在特定高稀疏度區間能實現更優的 Pareto 前沿**。透過深入刻畫預算配比（$\le 50\%$）、避開整數倍詞表的雜湊規則、嵌入放大（$\sqrt{D}$ 縮放與 Norm），以及與模型寬度和深度的幾何關係，我們成功駕馭了大規模嵌入擴展。

美團開源的 **LongCat-Flash-Lite** 證明了這一範式在工業級落地中的巨大威力：將 31.4B 參數賦予嵌入層，以僅約 3B 的激活計算量，在 Agent 與 Coding 領域達到了極具競爭力的性能水準，為下一代高效大模型架構設計開闢了全新方向。

---

## 譯者技術洞察與深度點評（Technical Insights）

### 1. 為什麼是美團 LongCat？與 DeepSeek、Kimi 架構路線的趣味對比
2025~2026 年的開源大模型架構呈現出百花齊放的局面：
* **DeepSeek-V3 / Kimi K3** 走的是「極致細粒度專家路由 + 深度通信重疊」路線（如 DeepSeek DualPipe、MoonEP），在 FFN 層將專家數量堆疊到數百個，激活數十個；
* **美團 LongCat 團隊** 則獨闢蹊徑：他們發現當 MoE 的專家數堆到一定程度後，邊際效益急劇遞減，同時顯存和通訊開銷暴增。因此，他們把超過 46% 的參數預算直接扔進了底層的 **N-gram Embedding**！
* 這種「寬淺主幹 + 超大局部記憶嵌入表」的做法，本質上是用極低成本的 $\mathcal{O}(1)$ 查表直接完成淺層語義綁定，讓後續僅有的 14 個 shortcut 層專注於高階邏輯與工具規劃，直接在 Agent 工具調用與程式碼修復（SWE-bench 54.4%）上打出了漂亮的成績。

### 2. 雜湊衝突與分詞器的「整數倍共振」陷阱
這篇論文最精彩的工程發現之一是圖 3b 中的**衝突異常尖峰**：
* 很多工程師直覺上認為雜湊表大小只要夠大、或是隨便挑個大質數或 $2^k$ 即可；
* 但美團團隊發現，BPE 分詞器的 Token ID 排布具有隱含的頻率結構，當 N-gram 詞表大小設為基礎詞表（如 128k）的整數倍時，會與多項式滾動雜湊發生諧波共振，導致衝突率反常飆升；
* 避開整數倍（例如選擇非倍數質數）能立竿見影地降低衝突，這是所有設計 Hash-based 模型的演算法工程師必須牢記的實戰細節。

### 3. 投機解碼：稀疏嵌入架構的最佳拍檔
傳統 MoE 在 Batch Size 很小時，顯存頻寬利用率低，難以發揮參數量大的優勢。而 LongCat 巧妙地將「激活參數小（僅 3B）」與「多步投機解碼」結合：
* 投機解碼天然放大了 Effective Batch Size；
* 透過專門的 N-gram Cache 與 CUDA 算子管理，成功在 8×H800 上實測斬獲 1.81 倍的吞吐量提升，完成了從理論稀疏到真實推理提速的閉環。
