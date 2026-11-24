---
title: Thinking Machines Lab 開源 Inkling 2 Mini 深度解析：用「迴圈深度」與「端到端 4-bit」把 26B 小模型壓出前沿 Agent 戰力
date: 2026-11-24 14:15:00
updated: 2026-11-24 14:15:00
categories: [AI, 論文與開源精讀]
tags: [Inkling-2-Mini, Thinking Machines Lab, Looped Transformer, MoE, NVFP4, Attention Residuals, DSpark]
description: 深度拆解 Thinking Machines Lab 開源的 26.4B 多模態迴圈 MoE 模型 Inkling 2 Mini：它如何透過中間 12 層 3 輪迴圈、Attention Residuals、128 選 8 專家剪枝與 1T 蒸餾修復、On-Policy 蒸餾對齊及端到端 NVFP4 訓練，在僅約 20 GB 的部署體積下取得與 27B 稠密模型匹敵的程式與搜尋 Agent 戰力。
mathjax: true
---

> **模型名稱**：Inkling 2 Mini  
> **開發機構**：Thinking Machines Lab (TML)  
> **發布日期**：2026 年 11 月 20 日  
> **授權條款**：Apache 2.0  
> **官方資源**：[TML 官方部落格](https://thinkingmachines.ai/news/inkling-2-mini/) / [TML 模型卡](https://thinkingmachines.ai/model-card/inkling-2-mini/) / [Hugging Face 倉庫](https://huggingface.co/thinkingmachines/Inkling-2-Mini)

---

## 前言：在筆電記憶體裡塞進「前沿 Agent 戰力」

Thinking Machines Lab（TML）於 2026 年 11 月 20 日正式開源了 **Inkling 2 Mini** —— 一款採用 Apache 2.0 授權、支援 **1M tokens 上下文**的多模態（文字＋圖片＋影片）稀疏 MoE 模型。

它的規格數字相當耐人尋味：總參數量為 **26.4B**，單次層訪問活躍參數僅 **3.4B**（計入 3 輪迴圈後相當於 **5.4B** 有效活躍參數）。更關鍵的是，由於模型從預訓練到強化學習（RL）前向全程採用 **NVFP4 + MXFP8** 混合低精度訓練，釋出的權重即為訓練精度，單一檢查點體積僅約 **19.5 GB（18.2 GiB）**，沒有任何轉檔量化損失，直接能在單張 32 GB 顯存 GPU、32 GB 以上 Apple Silicon 筆電或單台 NVIDIA DGX Spark 上本地運行。

在獨立評測機構 Artificial Analysis 的 Intelligence Index v5.0 中，Inkling 2 Mini 拿下 **20.1 分**，追平 27B 稠密模型 **Qwen3.8-27B (xhigh)**，但活躍參數少了約 5 倍，以「活躍參數 × 輸出 tokens」計算的推論計算量少了約 **2.8 倍**，每題平均花費從 $3.53 降到 **$0.59**（耗時從 38.2 分鐘縮短至 9.0 分鐘）。

<!-- more -->

深入拆解官方報告、Hugging Face `config.json` 與 safetensors 結構、NVIDIA NeMo AutoModel 文件以及 Goodfire 的可解釋性研究後，可以發現 **Inkling 2 Mini 幾乎沒有發明單一孤立的新元件，而是將 2018–2026 年間八組前沿技術精密組裝在一起**，並在「迴圈＋初始全關閘門」與「端到端 4-bit 訓練」上做出關鍵的工程整合。

---

## 一、整體架構總覽與參數拆解

![Inkling 2 Mini 架構圖](/higumalu-note/images/inkling-2-mini/architecture.png)

從 `config.json`（`model_type: "inkling2_loop"`）可直接驗證，Inkling 2 Mini 的語言骨幹共有 **24 層 Transformer**（隱藏維度 $d = 2304$），並切分為 4 個 6 層區塊（block）：

- **Prelude（前導區，第 0–5 層）**：執行 **1 次**。
- **Core（核心迴圈區，第 6–17 層，共 12 層）**：固定重複執行 **3 次**。
- **Coda（尾聲區，第 18–23 層）**：執行 **1 次**。

因此，每個 token 在單次前向傳播中會經歷 $6 + 12 \times 3 + 6 = 48$ 次層運算（layer visits）。在注意力層配置上，每 6 層採用 **5 層滑動視窗（SWA，window = 1,024）+ 1 層全域注意力**（第 5、11、17、23 層），這意味著 48 次層訪問中**只有 8 次是全域注意力**（第 5 層 1 次、第 11 與 17 層各 3 次、第 23 層 1 次）。

### 1.1 權重參數量拆解

根據 safetensors 實際張量形狀統計，26.39B 總參數中有高達 **93%** 集中在 MoE 專家層：

| 模組 | 參數量 | 佔比 / 備註 |
| --- | --- | --- |
| **MoE 專家層** | 24.461B | 約 92.7%（第 0–19 層 NVFP4，第 20–23 層 MXFP8） |
| **詞嵌入＋輸出層** | 0.926B | 有效詞表 200,058（padding 至 201,024），不共享權重 |
| **注意力層** | ~0.654B | 24 query / 8 KV heads（head dim 128）＋短卷積與閘門 |
| **視覺編碼器（ViT）** | 0.318B | 23 層 ViT（寬 1024）＋ 2×2 空間合併＋ Projector |
| **Router** | 0.007B | 每層 128 專家線性層＋ FP32 bias |
| **合計** | **26.389B** | 官方標示 26.4B（每多一輪迴圈約增加 1.09B 活躍參數） |

### 1.2 視覺編碼器與 Tokenizer 細節

- **不縮放的原生視覺輸入**：視覺骨幹為 23 層 ViT（16 heads、MLP 4096、2D RoPE + 32×32 插值絕對位置編碼）。圖片不強制縮放至固定解析度，以 `20×20` patch 切分（時間軸複製成 2 幀以相容影片）並做 `2×2` 空間合併，最終每 **40×40 像素對應 1 個語言 token**，建議輸入邊長為 40–4,096 px。
- **Tokenizer 與推理強度控制**：採用 tiktoken 格式（推論為 `o200k` 系列），支援 `none (0.0)`、`minimal (0.1)`、`low (0.2)`、`medium (0.7)`、`high (0.9，預設)` 到 `xhigh/max (0.99)` 等連續推理強度控制（透過在首則非 system 訊息前插入 `Thinking effort level: X` 觸發）。

---

## 二、八大核心技術深度拆解

### 1. 迴圈（遞迴深度）Transformer 與「初始全關」跨輪閘門

從 2018 年 Google 的 [Universal Transformers](https://arxiv.org/abs/1807.03819)、2025 年的 [Huginn](https://arxiv.org/abs/2502.05171) 到 [Ouro / LoopLM](https://arxiv.org/abs/2510.25741)，遞迴深度架構的核心直覺都是：**用「重複計算」代替「增加參數」來換取深度**。

不過，[Saunshi 等人（2025）](https://arxiv.org/abs/2502.17416) 與 Ouro 的實驗已證實一個關鍵限制：**迴圈只幫助推理，幾乎不增加知識記憶容量**。Ouro 實測顯示迴圈 1 次與 4 次的模型知識容量皆維持在約 2 bits/參數；但免 CoT 的多步模運算與形式邏輯準確率卻能暴增逾百個百分點。

Inkling 2 Mini 在這條路線上做了兩項關鍵工程設計：

1. **為什麼迴圈「中間 12 層」而非最後 12 層？**  
   TML 官方消融顯示，迴圈中間 12 層在 28 項評測中有 **27 項**優於迴圈最後 12 層。訓練到第 2,000 步時，不迴圈基線需要 **1.51–1.91 倍**計算量才能追上中間迴圈的 loss（對最後層迴圈則為 1.24–1.36 倍）；在 512K-token 長代理軌跡上，非迴圈模型更需多耗費 **140%** 推論算力才能達到同等 loss。
2. **可變與漸進引入的 Rank-64 跨輪閘門**  
   為了讓迴圈能在預訓練中途平滑引入而不破壞原本特徵分佈，Core 層在第 2、3 輪會讀取上一輪區塊 2 與區塊 3 的輸出總和作為跨輪訊息 $m_s$（分別注入 Q、K、V 與 MoE 四個讀取點），並經過一個 Rank-64 控制器：

$$z = P\,[\mathrm{RMSNorm}(x);\ \mathrm{RMSNorm}(m)], \quad P \in \mathbb{R}^{64 \times 2d}$$

$$\kappa = 1 - \tanh(Fz), \quad \gamma = \tanh(Gz), \quad x' = \kappa \odot x + \gamma \odot m$$

訓練初期權重設定使 $\gamma \approx 0$、$\kappa \approx 1$（閘門初始全關，完全擋住跨輪訊息），再隨訓練逐步打開。

此外，TML 同時發布了研究變體 **TML-d24-varloop-research**（訓練時以 `1:2:6:2:1` 比例取樣 1–5 輪迴圈），證實迴圈次數越多，模型越能把思考壓進隱狀態裡：從 1 輪增加到 3 輪時，無 CoT 時間視界（Think Fast benchmark）從 **16 秒提升到 37 秒**，而 GPQA Diamond 所需的顯式推理 token 數則從 **3,703 降到 1,731**。

---

### 2. Attention Residuals（AttnRes）：告別單一 Residual Stream

傳統 PreNorm Transformer 使用 $x_{l+1} = x_l + f(x_l)$ 累加殘差，層數加深後早期層的訊號會被稀釋。Inkling 2 Mini 採用了 Kimi Team 於 2026 年 3 月發表的 [Block Attention Residuals](https://arxiv.org/abs/2603.15031) 概念：

- **以 6 層為一個 Block**：每個層（包含注意力與 MoE 子層）不再只吃上一層的殘差流，而是透過一個 FP32 的可學習查詢向量 $q$，對前面已完成各 Block 輸出的 RMSNorm 結果做 softmax 加權聚合：

$$x = \sum_j \alpha_j \hat{v}_j, \quad \alpha_j = \mathrm{softmax}_j\!\left(\frac{q^\top \hat{v}_j}{\sqrt{d}}\right), \quad \hat{v} = \frac{v}{\mathrm{RMS}(v)}$$

- **全域層與讀取點**：從權重結構可驗證全模型共有 **56 個讀取點**（23 個 `attn_queries`，其中 4 個全域層為 Q/K/V 分開讀取；24 個 `mlp_queries`；1 個 `output_query`）。由於查詢向量與輸入無關（僅對 Block 總和計算），它能兼顧低 I/O 開銷（Block 版每層每 token 僅 5.5d，大幅低於 mHC 的 34d）與極高的深度表達力。

---

### 3. 支撐 1M 上下文的注意力組合拳

為了在 1M tokens 長上下文下壓低 KV Cache 並避免注意力崩潰，Inkling 2 Mini 疊加了五項機制：

1. **5:1 混合滑動視窗＋GQA**：每 6 層只有 1 層全域注意力（第 5、11、17、23 層），其餘 20 層為 `window=1,024` 的滑動視窗；搭配 24Q / 8KV GQA。以 BF16 估算，1M tokens 下全域層 KV cache 約 32 GiB、滑動視窗層固定僅約 0.16 GiB，比全全域注意力節省約 **6 倍**記憶體。
2. **捨棄 RoPE，改用 Learned Relative Position Logits**：語言骨幹完全不使用 RoPE（僅 DSpark drafter 使用），而是計算低維相對位置特徵（`d_rel=16`, `rel_extent=1024`, `rel_l2_clip=8.0`）加上距離偏置；vLLM 甚至為此專門撰寫了帶 sheared-bias 的 FlashAttention-4 kernel。
3. **Learned Zero-Valued Virtual Sink**：每頭依輸入算出一個 sink logit，對應 value 恆為 0 的虛擬 token，專門用來吸收無關的注意力機率質量。
4. **Gated Attention（輸出閘門）**：在 SDPA 輸出後加入逐頭逐通道的 `gate_attn` 閘門（源自 Qwen 2025 研究，可進一步減輕注意力尖峰並改善長上下文外推）。
5. **每層 4 個短卷積（Short Conv1d, kernel=4）**：分別作用於 K、V、注意力輸出與 MoE 輸出。20 個滑動層使用靜態 depthwise causal conv1d，而 **4 個全域層的 K/V 則由 token 動態產生 4 個 taps**（Dynamic Short Convolutions）。

---

### 4. 細粒度 MoE 與「先剪 2/3 再用 1T Tokens 蒸餾修復」

Inkling 2 Mini 採用細粒度 MoE 設計：**128 個路由專家、每 token 啟用 top-8、無共享專家**（`n_shared_experts: 0`），專家隱藏中間寬度為 1,152。Router 使用 sigmoid 分數搭配每個專家一個 FP32 可訓練校正 bias（而非傳統輔助負載平衡 loss），激活函數則為 **capped SwiGLU**（gate 截斷在 4.0，up 截斷在 25.0）。

更特別的是它的**專家剪枝策略**：
- **預訓練起點**：最初以每層 **384 個專家（稀疏度 48）** 訓練。
- **中期激進剪枝**：直接在訓練中期依路由使用頻率**剪掉 2/3（66.7%）的專家**，只保留最常用的 128 個。
- **1T Tokens 蒸餾修復（Healing）**：[REAP 論文（2025）](https://arxiv.org/abs/2510.13999) 曾指出不重訓時單靠頻率剪枝會讓模型徹底崩潰（例如 Kimi-K2 剪 50% 後程式評測從 0.631 暴跌至 0.056）。TML 的解法是直接用一個更大的內部 Inkling 老師模型進行 **1T tokens 的蒸餾修復**（規模約 NVIDIA [Minitron](https://arxiv.org/abs/2407.14679) 的 10 倍），讓剩下的 128 個專家吸收被刪除專家的能力。

---

### 5. 後訓練配方：RL 學能力，On-Policy 蒸餾做對齊

Inkling 2 Mini 的後訓練分為三階段：**SFT → RL（2.2T tokens、2,000 步）→ On-Policy 蒸餾對齊**。

TML 在其技術部落格指出：傳統 RL 每個 episode 只能提供約 $O(1)$ bits 的稀疏獎勵回饋，算力成本極高；而 **On-Policy 蒸餾**（源自 Google DeepMind 的 [GKD](https://arxiv.org/abs/2306.13649)）讓學生自己採樣軌跡，由老師模型對每個 token 計算一次 logprob，直接把 RL 的 advantage 設為**負的逐 token reverse KL**：

$$\text{advantage}_t = -\left(\log \pi_{\text{student}}(y_t \mid x, y_{<t}) - \log \pi_{\text{teacher}}(y_t \mid x, y_{<t})\right)$$

這種「一行程式碼」接上 RL importance-sampling loss 的做法，讓每個 token 都擁有密集回饋（$O(N)$ bits），同時保有 on-policy 避免 exposure bias 與 mode-seeking（unhackable）的優勢，以極低成本完成最終對齊。

---

### 6. 端到端 NVFP4 / MXFP8 低精度訓練與硬體協同

為了突破低精度訓練常在轉換或量化後掉點的瓶頸，TML 採用了與部署完全一致的原生低精度配方（[End-to-End 4-Bit Training](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/)）：

- **前向與反向分離**：從預訓練到 RL，**第 0–19 層專家層前向皆為 NVFP4**（E2M1 元素、block size 16、FP8 E4M3 區塊縮放＋每專家/每 token FP32 外層縮放）；**最後 1/6 的層（第 20–23 層）保留 MXFP8**；反向傳遞則統一使用 MXFP8＋隨機捨入（Stochastic Rounding）。
- **Four Over Six 與確定性 Kernel**：在激活值量化上結合 [Four Over Six](https://arxiv.org/abs/2512.02010) 技巧減緩 FP4 最大值區間（66.6%–100%）的量化誤差，MoE combine 通訊也透過隨機 Hadamard 轉換壓縮至 8.25 bit。
- **訓練即部署**：因為訓練終點即是 NVFP4 + MXFP8 權重，發布檢查點（`quant_method=modelopt`）**不需額外 PTQ 校準即可直接上線**，在 GB300 上預訓練比全 MXFP8 快 8%、RL 快 20%，獎勵表現無可見差異。

---

### 7. DSpark 推測解碼與本地部署生態

Inkling 2 Mini 附帶了一個獨立的 636 MB 推測解碼草稿模型（`mtp.safetensors`），採用 [DSpark（2026-07）](https://arxiv.org/abs/2607.05147) 架構：
- 由 **3 層 decoder**（全模型唯一使用 RoPE 的部分）組成，讀取目標模型第 **5、23、41 次層訪問**（恰好對應 Prelude 末、Core 第 2 輪第 11 層、Core 第 3 輪第 17 層全域層）以及最終狀態（第 48 次訪問）。
- 平行 Backbone 一次提出 `block_size=8` 個草稿 token，再透過一個 rank-256 的 GRU 式 `markov_head` 補回 block 內部的 token 相依性，並由信心 head 動態決定每次驗證長度。
- **實測加速**：Unsloth 在 B200 上實測開啟 DSpark 後，greedy 解碼從 **99.9 tok/s 提升至 117.3 tok/s**；在 DGX Spark 上透過 Stanford 的 Intelligence per Watt 指標實測達到 **0.703**（遠高於 Qwen3.6-35B-A3B 的 0.430）。
- **部署限制提醒**：由於 NVFP4 專家權重不可做 Tensor Parallel (TP) 切分，在 SGLang 上部署時 **EP 必須等於 TP**；SGLang 同時提供 `--inkling2-loop-share-cache` 選項讓三輪迴圈共用 KV cache，可使 KV pool 空間翻倍（實測 gsm8k 甚至從 88.5% 微升至 90.0%）。

---

### 8. Goodfire 可解釋性研究：迴圈裡真的在「分段演算」

可解釋性團隊 [Goodfire](https://www.goodfire.com/research/interpreting-inkling-2-mini/) 對 Inkling 2 Mini 進行了權重空間與激活分析，揭示了迴圈模型的內部運作機制：

- **Writer 集中在 Block 末層**：跨層寫入主要集中在第 6、12、18 層，而第 18 層在第 1、2 輪與第 3 輪之間出現顯著行為反轉——巨大離群通道 $d_{2027}$（佔平方範數高達 46%）在第 1、2 輪呈正向尖峰，第 3 輪反轉為負向尖峰，扮演「最後一輪收尾清理」的角色。
- **Logit-lens 驗證**：傳統 logit-lens 只在每一輪的末層有效，代表中間層確實專注於內部隱狀態轉換，而非提早收斂到詞表空間。

---

## 三、基準測試表現與獨立評測

Inkling 2 Mini 最強的領域集中在**程式代理（Agentic Coding）、搜尋與工具調用、長上下文與桌面 GUI 定位**。以下節選官方模型卡數據與對比模型（粗體為該列最佳）：

| 類別 | 評測項目 | Inkling 2 Mini<br/>(5.4B/26.4B) | Qwen3.8-27B<br/>(27B Dense) | Gemma 4 31B<br/>(30.7B Dense) | Nemotron 3.5 Super VL<br/>(12B/124.5B) |
| --- | --- | --- | --- | --- | --- |
| **綜合** | AA Intelligence Index v5.0 | **20.1** | **20.1** | 4.4 | 13.6 |
| **程式** | SWE-bench Verified (%) | **79.2** | 77.8 | 46.1 | 71.1 |
| **程式** | SWE-bench Pro v2 Hard (%) | **77.9** | 73.8 | 14.2 | 44.1 |
| **程式** | Vibe Code Bench v1.1 (%) | **77.9** | 64.9 | – | 3.5 |
| **程式** | Terminal Bench v2.1 (%) | 71.6 | **73.0** | 43.4 | 60.7 |
| **搜尋/工具** | BrowseComp (%) | **79.0** | 61.6 | 37.0 | 78.9 |
| **搜尋/工具** | DeepSearchQA (%) | **85.3** | 79.7 | 61.7 | 77.7 |
| **搜尋/工具** | MCP Atlas (%) | **82.7** | 77.3 | 54.2 | 67.7 |
| **視覺** | ScreenSpot Pro (%) | **76.5** | 57.6 | 5.9 | 43.5 |
| **視覺** | OmniDocBench v1.6 (%) | **92.8** | 91.5 | 82.5 | 77.9 |
| **推理** | ARC-AGI-2 (%) | **49.3** | 42.4 | – | 1.5 |
| **推理** | GPQA Diamond (%) | 86.6 | **90.5** | 85.7 | 89.5 |
| **長上下文** | LongBench v2 (%) | **66.9** | 52.3 | 52.9 | 62.1 |
| **長上下文** | MRCR v2 256K–512K (%) | **46.4** | 38.0 | – | 5.7 |
| **知識** | AA-Omniscience 非幻覺率 (%) | 63.4 | **71.4** | 14.5 | 69.5 |
| **安全** | 破壞性動作（內部，越低越好） | **10.2%** | 35.2% | 75.8% | 65.6% |

**獨立第三方觀察（Artificial Analysis）**：  
Inkling 2 Mini 每題平均產生約 **152K 輸出 tokens**，比 Qwen3.8-27B（約 110K）多出 38%——這是它用「較長推理＋迴圈深度」換取高分代價的展現。但因為活躍參數只有對手的約 1/5，整體計算量與推論成本仍具備壓倒性優勢，穩居「智慧 vs 活躍參數」的 Pareto 前緣。

---

## 四、限制、未公開資訊與技術點評

### 官方自述的實務限制
1. **事實知識與幻覺**：印證了「迴圈不增知識」的理論，Inkling 2 Mini 在事實回憶類任務落後大型模型與同級稠密模型（AA-Omniscience 63.4% vs Qwen3.8-27B 71.4%），且在不知道答案時較容易自信地說錯。
2. **推理迴圈與代理失誤**：在除錯程式時容易陷入過長的推理迴圈（且就算把推理強度設為最低 `none` 也無法完全關閉推理），建議生產環境務必設定輸出 token 與 Agent 回合上限；此外在 Agent 操作中有未經確認就刪除／覆寫檔案的紀錄。
3. **中文與非英文表現**：官方明確指出非英文品質較低，中文測試會出現「措辭簡略、中英混雜」的現象。
4. **CoT 可監控性**：因部分推理移入看不見的迴圈隱狀態，官方特別做了安全性檢查，結果顯示 `CoT-Control-QA` 遵從率最多僅 **0.1%**，LinuxArena 隱蔽任務成功率在可看思考內容時只有 **1.3%**，目前尚未發現隱蔽失控的跡象。

### 尚未公開的技術拼圖
目前官方仍未揭露的資訊包含：預訓練總 token 數與詳細資料配比、知識截止日、1M 上下文具體的長距離延展策略、SFT/RL 獎勵函數設計、實際訓練算力規模，以及 `log_scaling_alpha=0.1` 等部分細微公式實作。此外不同文件對活躍參數（3B / 3.4B / 3.5B / 5.4B）與研究版 `MRCR 512K-1M`（37.7% vs 81.4%）的口徑存在差異，閱讀時需留意定義。

---

### 🐻 小熊技術點評

Inkling 2 Mini 給了開源小模型領域一個非常清晰的工程啟示：**當單機記憶體（20–32 GB）卡死總參數量上限時，「知識」可以靠外部搜尋與工具（MCP / BrowseComp）補齊，但「代理推理與代碼規劃深度」必須靠架構創新硬擠出來。**

TML 的解法非常務實：
1. 先用 **384 專家的大模型**學出足夠強的表徵，再用 **剪枝 2/3＋1T tokens 蒸餾**硬壓回 26.4B；
2. 用 **中間 12 層 × 3 輪迴圈＋AttnRes** 在不增加權重體積的前提下換取 48 層深度；
3. 用 **訓練即部署的 NVFP4** 省去 4-bit 量化掉點的風險，直接把 26B 模型壓進筆電級記憶體。

雖然它在純事實知識與中文流暢度上還有明顯妥協，但作為一款能跑在個人工作站或筆電上的 **Apache 2.0 程式與工具代理模型**，Inkling 2 Mini 已經把「小體積高智慧」的工程標竿推上了一個全新台階。

---

## 參考資料

- [Introducing Inkling 2 Mini（Thinking Machines Lab 部落格）](https://thinkingmachines.ai/news/inkling-2-mini/)
- [Inkling 2 Mini 模型卡（TML）](https://thinkingmachines.ai/model-card/inkling-2-mini/) / [Hugging Face 模型卡](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
- [Welcome Inkling 2 Mini（Hugging Face 部落格）](https://huggingface.co/blog/inkling2mini)
- [Artificial Analysis 獨立評測報告](https://artificialanalysis.ai/articles/inkling-2-mini)
- [TML: End-to-End 4-Bit Training](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/) / [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Goodfire: Interpreting Inkling 2 Mini](https://www.goodfire.com/research/interpreting-inkling-2-mini/)
- 核心參考論文：[Universal Transformers](https://arxiv.org/abs/1807.03819)、[Huginn](https://arxiv.org/abs/2502.05171)、[Ouro / LoopLM](https://arxiv.org/abs/2510.25741)、[Attention Residuals](https://arxiv.org/abs/2603.15031)、[REAP](https://arxiv.org/abs/2510.13999)、[Minitron](https://arxiv.org/abs/2407.14679)、[GKD](https://arxiv.org/abs/2306.13649)、[NVFP4 預訓練](https://arxiv.org/abs/2509.25149)、[Four Over Six](https://arxiv.org/abs/2512.02010)、[DSpark](https://arxiv.org/abs/2607.05147)、[Intelligence per Watt](https://arxiv.org/abs/2511.07885)
