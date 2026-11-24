---
title: Inkling 2 Mini 架構與工程實作解析：Looped MoE、Block Attention Residuals 與原生 NVFP4 訓練
date: 2026-11-24 14:15:00
updated: 2026-11-24 15:28:00
categories: [AI, 論文與開源精讀]
tags: [Inkling-2-Mini, Thinking Machines Lab, Looped Transformer, MoE, Attention Residuals, NVFP4, Four Over Six, DSpark, vLLM, SGLang]
description: 綜合 Hugging Face config.json、safetensors 權重形狀、NVIDIA NeMo AutoModel 文件與 Goodfire 可解釋性報告，完整拆解 Inkling 2 Mini 的中間 12 層 3 輪迴圈架構、Rank-64 跨輪閘門、Block Attention Residuals、1M 上下文注意力與 KV Cache VRAM 估算、384→128 專家剪枝與 1T 蒸餾修復、端到端 NVFP4/Four Over Six 訓練，以及 DSpark 推測解碼與多框架部署實測。
mathjax: true
---

> **模型名稱**：Inkling 2 Mini（`model_type: "inkling2_loop"`）  
> **發布機構**：Thinking Machines Lab (TML)  
> **發布日期**：2026-11-20  
> **授權**：Apache 2.0（另附 [Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/)）  
> **主要來源**：[TML 官方部落格](https://thinkingmachines.ai/news/inkling-2-mini/) / [TML 模型卡](https://thinkingmachines.ai/model-card/inkling-2-mini/) / [HF 倉庫與 config.json](https://huggingface.co/thinkingmachines/Inkling-2-Mini) / [Goodfire 可解釋性報告](https://www.goodfire.com/research/interpreting-inkling-2-mini/)

---

## 概述：定位與核心架構取捨

Thinking Machines Lab（TML）於 2026 年 11 月 20 日釋出的 **Inkling 2 Mini** 是一個多模態（文字＋圖片）稀疏 MoE 迴圈 Transformer。官方目前**沒有發表正式技術論文**，本文綜合 TML 部落格、模型卡、Hugging Face `config.json` 與 `safetensors` header 張量形狀、NVIDIA NeMo AutoModel `inkling2` 文件、vLLM/SGLang/Unsloth 部署文件以及 Goodfire 的可解釋性研究，還原其架構設計、數學形式與系統工程細節。

從 VRAM 佔用與計算量的折衷來看，Inkling 2 Mini 將主權重體積壓在 **18.2 GiB（19.5 GB）**（另附 636 MB 的 `mtp.safetensors` drafter），使其可完整載入單張 32 GB VRAM GPU（如 RTX 5090）、NVIDIA DGX Spark 或 32 GB 以上統一記憶體的 Apple Silicon，並透過三個正交的工程手段拉高推理與長程代理能力：

1. **以重複計算換取有效深度**：24 層實體解碼層切為 Prelude（0–5）、Core（6–17）、Coda（18–23），中間 12 層共用權重執行 3 輪，每個 token 實際經過 48 次層運算，以 26.4B 總參數、3.4B 單輪活躍參數換取 5.4B 有效活躍計算量。
2. **中期大規模剪枝＋蒸餾修復**：預訓練從每層 384 專家（top-8、稀疏度 48）起步，中期依啟用頻率剪掉 2/3 僅留 128 專家，再以更大的內部 Inkling 模型蒸餾 1T tokens 修復。
3. **訓練與部署數值一致的原生 NVFP4**：第 0–19 層專家從預訓練到 RL 前向全程採用 NVFP4（激活值搭配 Four Over Six 自適應縮放），第 20–23 層維持 MXFP8，釋出權重即訓練精度，不需額外 PTQ 校準。

<!-- more -->

本文依序拆解：`config.json` 與張量形狀、Looped Transformer 與跨輪閘門、Block Attention Residuals、單層前向資料流與 1M KV Cache VRAM 估算、MoE 剪枝與 On-Policy 後訓練、NVFP4 + Four Over Six 低精度訓練、DSpark 推測解碼與框架實測，最後整理 Goodfire 可解釋性發現、評測對照與尚未公開的實作缺口。

---

## 一、架構規格、`config.json` 與參數拆解

![Inkling 2 Mini 架構：中間 12 層共用權重執行 3 輪](/higumalu-note/images/inkling-2-mini/architecture.png)

### 1.1 `config.json` 核心欄位驗證

從 HF 倉庫的 `config.json`（`architectures: ["Inkling2LoopForConditionalGeneration"]`、`dtype: bfloat16`、`eos_token_id: 200006`）可直接驗證全部結構超參數：

| 類別 | `text_config` / 子配置欄位與數值 |
| --- | --- |
| **骨幹** | `num_hidden_layers=24`、`hidden_size=2304`、`model_max_length=1048576`（1M） |
| **迴圈 (`loop_config`)** | `core_start=6`、`core_end=18`、`num_passes=3`、`gate_rank=64`、`is_varloop=false` |
| **詞表** | `unpadded_vocab_size=200058`、`vocab_size=201024`（embed 與 unembed 不共享權重） |
| **注意力** | `num_attention_heads=24`、`num_key_value_heads=8`、`head_dim=128`、`q_bias=false`、`o_bias=false` |
| **滑動視窗與層型** | `sliding_window_size=1024`、`local_layer_ids=[0..4, 6..10, 12..16, 18..22]`（全域層為 5, 11, 17, 23） |
| **相對位置與截斷** | `d_rel=16`、`rel_extent=1024`、`rel_l2_clip_threshold=8.0`、`qk_l2_clip_threshold=22.627...`（$=\sqrt{512}$） |
| **長度縮放與 Norm** | `log_scaling_n_floor=8192`、`log_scaling_alpha=0.1`、`rms_norm_eps=1e-6`、`use_value_norm_no_gamma=true` |
| **短卷積與 AttnRes** | `sconv_kernel_size=4`、`n_layers_per_attn_res_block=6`、`attn_res_normalize_values=true` |
| **MoE** | `n_routed_experts=128`、`num_experts_per_tok=8`、`n_shared_experts=0`、`intermediate_size=1152`、`route_scale=8.0`、`use_gate_bias=true` |
| **Softcap** | `mlp_gate_softcap=4.0`、`mlp_up_softcap=25.0`、`final_logit_softcapping=30.0` |
| **視覺 (`vision_config`)** | `num_hidden_layers=23`、`hidden_size=1024`、`num_attention_heads=16`、`mlp_ratio=4.0`、`native_patch_size=20`、`temporal_patch_size=2`、`adapter_merge_size=2`、`projection_hidden_dim=4096`、`posemb_grid_size=32` |
| **Drafter (`mtp_config`)** | `num_hidden_layers=3`、`intermediate_size=9216`、`rope_theta=1e6`、`local_layer_ids=[0,2]`、`block_size=8`、`markov_rank=256`、`markov_head_type="rnn_hidden"`、`target_layer_ids=[5,23,41,48]` |
| **量化 (`quantization_config`)** | `quant_method="modelopt"`、`quant_algo="NVFP4"` |

### 1.2 參數量拆解與存儲精度

依 `safetensors` header 張量形狀逐項加總，26.389B 總參數中有 **92.7%** 集中在 MoE 專家權重：

| 模組 | 參數量 | 存儲精度與張量說明 |
| --- | --- | --- |
| **MoE 專家層** | 24.461B | 第 0–19 層 NVFP4（`U8` 封裝兩個 fp4＋FP8 E4M3 scale＋每專家 amax/outer scale）；第 20–23 層 MXFP8（`F8_E4M3`＋`E8M0` scale） |
| **詞嵌入＋輸出層** | 0.926B | `[201024, 2304]` 各一份，BF16，不共享權重 |
| **注意力與短卷積** | ~0.654B | BF16（含 `wq_du`, `wk_dv`, `wv_dv`, `wr_du`, `gate_attn`, `wo_ud`, `attn_sink_proj`, `sconv` 等） |
| **視覺編碼器** | 0.318B | 23 層 ViT（backbone ~0.29B，與 Inkling 2 Small 292M Perception Encoder 契約一致）＋ Projector |
| **Router 與 AttnRes** | 0.007B | `gate.weight` `[128, 2304]` BF16、`gate.bias` `[128]` F32、AttnRes 查詢向量 F32 |
| **合計** | **26.389B** | 主權重檔 **19.5 GB（18.2 GiB）**；另附 `mtp.safetensors` 636 MB |

每多一輪 Core 迴圈涉及 12 層的（8/128 專家＋注意力）運算，約增加 **1.09B** 活躍參數，因此 1 / 3 / 5 輪對應的有效活躍參數分別為 **3.4B / 5.4B / 7.4B**（Artificial Analysis 另統計單文字 token 觸及的唯一權重為 3.1B）。

### 1.3 原生解析度 ViT 管線與 Tokenizer 協議

- **不強制縮放的視覺編碼**：圖片不預先 resize（RGBA 先合成至棋盤格、padding 填 `-1`、mean/std 0.5、建議邊長 40–4,096 px），切為 `20×20` patch 並沿時間軸複製成 2 幀以兼容影片，經 patch embedding（`1200 → 1024`）與 `32×32` 可學絕對位置網格（插值）+ 2D RoPE 進入 23 層 ViT（16 頭、GELU MLP 4096、pre-LN、LayerScale、CLS）。輸出經 `2×2` 空間合併（`4096`）與無 bias Projector（`4096 → 4096 → 2304`），直接插入 `embed_norm` 之後且**不再做正規化**（NeMo 文件確認）。每 **40×40 像素對應 1 個 token**（token 數 $\approx \lceil H/40 \rceil \times \lceil W/40 \rceil$），每張圖在 ViT 內獨立做注意力，圖片間不混算。
- **Tokenizer 與推理強度協議**：`PreTrainedTokenizerFast` + tiktoken 格式（`199999` 為 `<|endoftext|>`，推論為 `o200k` 系列），有效詞表 200,058，padding 至 201,024。`chat_template.jinja` 在首則非 system 訊息前插入 `Thinking effort level: X`（`none=0`, `minimal=0.1`, `low=0.2`, `medium=0.7`, `high=0.9` 預設, `xhigh/max=0.99`，或直接給 0–0.99 數值）。工具宣告走 `<|message_system|>tool_declare<|content_xml|>` + 排序緊湊 JSON，工具呼叫走 `<|content_invoke_tool_json|>`（args 必須為 dict），不接受音訊且只有 user/tool 訊息可含圖片。

---

## 二、Looped Transformer：中間 12 層迴圈與 Rank-64 跨輪閘門

### 2.1 與 Huginn、Ouro / LoopLM 的結構對照

遞迴深度（Recurrent Depth）架構從 2018 年 [Universal Transformers](https://arxiv.org/abs/1807.03819) 演進至 2025–2026 年，主要形成三種實作範式：

| 維度 | [Huginn](https://arxiv.org/abs/2502.05171) (2025) | [Ouro / LoopLM](https://arxiv.org/abs/2510.25741) (2025–2026) | Inkling 2 Mini (2026) |
| --- | --- | --- | --- |
| **層結構** | Prelude 2 / Core 4 / Coda 2 | 無 Prelude/Coda，整組 24 層重複 | Prelude 6 / Core 12 / Coda 6 |
| **跨輪注入** | 每步 $[s_{i-1}; e]$ 串接→ adapter | 無 input injection | 每輪從 Prelude 輸出 $e$ 出發＋上一輪訊息經 Rank-64 閘門 |
| **迴圈次數** | 訓練時 log-normal Poisson 抽樣（平均 32） | 最多 4 步，exit gate $\lambda_t$ 決定退出 | 固定 3 輪（研究版 `varloop` 為 1–5 輪） |
| **殘差與正規化** | Sandwich norm | Sandwich RMSNorm | Block Attention Residuals |
| **反向傳播** | 僅對最後 8 步截斷 | 全部展開 | 未公開 |
| **KV Cache** | 可共享（$i \bmod k$） | Decode 只存最後一步幾乎無損 | 48 次訪問各有一份（SGLang 可選共用） |
| **規模** | 3.5B Dense | 1.4B / 2.6B Dense | 26.4B MoE（5.4B 有效活躍） |

**為何迴圈「中間 12 層」且固定 3 輪？**
- **中間層 vs 最後層**：TML 消融實驗顯示，迴圈中間 12 層在 28 項評測中有 **27 項**優於迴圈最後 12 層。訓練至第 2,000 步時，非迴圈基線需要 **1.51–1.91×** 計算量才能追平中間迴圈的 loss（對最後層迴圈僅 1.24–1.36×）。在未剪枝對照中，非迴圈模型訓練算力雖少 7.5%，但在 512K-token 代理軌跡上需多耗費 **140%** 推論算力才能達到同等 loss。保留 6 層 Coda 讓模型在潛空間迴圈演算完後，有專屬層數將表徵映射回詞表空間。
- **固定 3 輪 vs 動態退出**：Ouro 論文記錄其 DAPO/GRPO RL 階段未能超過 SFT，主因是逐 token 動態深度與 vLLM 固定執行圖不相容。Inkling 採用固定 3 輪，使 48 次層訪問可視為一般靜態計算圖，完整相容 RL 訓練管線、CUDA graph 與 prefix caching。

### 2.2 跨輪訊息與「初始全關」Rank-64 閘門

跨輪機制僅在第 2、3 輪的 Core 層（第 6–17 層）啟用，每層針對 $s \in \{Q, K, V, \text{MoE}\}$ 四個讀取點各配一組參數（權重 `middle_cycle_cross_pass_queries` 與 `middle_cycle_cross_pass_kv_queries`，各 `[24, 2304]`）。先以可學查詢 $u_s$ 對上一輪區塊 2（第 6–11 層）與區塊 3（第 12–17 層）的輸出總和做加權讀取：

$$m_s = \sum_{j \in \{2,3\}} \beta_j \hat{v}_j^{(t-1)}, \qquad \beta_j = \mathrm{softmax}_j\!\left(\frac{u_s^\top \hat{v}_j^{(t-1)}}{\sqrt{d}}\right), \qquad \hat{v} = \frac{v}{\mathrm{RMS}(v)}$$

再透過 Rank-64 控制器（`controllers.{attn,k,v,mlp}`：`down_weight` `[64, 4608]`、`forget_weight` `[2304, 64]`、`gain_weight` `[2304, 64]`）混合當前輸入 $x$ 與跨輪訊息 $m_s$（Goodfire 依權重重建）：

$$z = P\,[\mathrm{RMSNorm}(x);\ \mathrm{RMSNorm}(m_s)], \qquad P \in \mathbb{R}^{64 \times 2d}$$

$$\kappa = 1 - \tanh(Fz) \in (0, 2), \qquad \gamma = \tanh(Gz) \in (-1, 1), \qquad x' = \kappa \odot x + \gamma \odot m_s$$

這個設計的工程關鍵在於**初始化使 $\gamma \approx 0,\ \kappa \approx 1$**：在訓練中途將非迴圈模型切換為迴圈模型時，閘門完全封鎖跨輪訊息，每輪初始行為等價於原非迴圈模型，避免破壞既有特徵分佈，再隨梯度更新逐步打開。

### 2.3 迴圈的能力邊界：增推理、不增知識容量

[Saunshi 等人 (2025)](https://arxiv.org/abs/2502.17416) 與 Ouro 的控制實驗清楚劃出迴圈架構的邊界：
- **知識容量不隨迴圈增加**：Ouro Capo 實驗中，loop-1 與 loop-4 皆維持約 **2 bits/param**；Saunshi 等人的 1B 對照中，24×1 閉卷 QA 為 11.2，而 4×6 迴圈反降至 6.7。
- **知識操作與多步推理顯著受益**：Ouro Mano（$\mathbb{F}_{23}$ 無中間步驟前序表達式）中，2 層迴圈 6 次達 **98.1%**（$L=10$），同 2 層不迴圈僅 **21.5%**；MMLU 中基礎數學 +155.6%、形式邏輯 +143.3%，而 global facts 僅 +8.3%。

TML 同步釋出的可變迴圈研究版 **TML-d24-varloop-research**（與 Mini 共用預訓練與中期第一階段，之後以 `1:2:6:2:1` 比例取樣 1–5 輪，後訓練固定 3 輪）直接驗證了這一點：

| 評測項目 | 1 輪 (3.4B) | 3 輪 (5.4B) | 5 輪 (7.4B) | Inkling 2 Mini (固定 3 輪) |
| --- | --- | --- | --- | --- |
| **SWE-bench Verified (%)** | 56.3 | 78.7 | 79.6 | 79.2 |
| **ARC-AGI-2 (%)** | 0.8 | 22.1 | 27.1 | 49.3（研究版頁面寫 51.3） |
| **MRCR 512K–1M (%)** | 20.6 | 81.4 | 79.9 | 37.7 |
| **GPQA 平均推理 tokens** | 3,703 | 1,731 | 1,661 | — |
| **Think Fast 無 CoT 時間視界** | 16 秒 | 37 秒（2 輪 26 秒） | — | — |

1 輪到 3 輪收益最大，3 輪後趨於飽和；同時增加迴圈可將顯式 CoT 長度壓縮超過一半（3,703 → 1,731 tokens），並將 [Think Fast](https://arxiv.org/abs/2606.07157) 的無 CoT 時間視界由 16 秒拉長至 37 秒。

---

## 三、Block Attention Residuals：取代單一 Residual Stream

傳統 PreNorm 殘差 $h_{l+1} = h_l + f_l(h_l)$ 以固定權重 1 累加，深度增加時隱藏狀態範數膨脹、單層貢獻被稀釋。Inkling 2 Mini 採用 Kimi Team 於 2026 年 3 月提出的 [Block Attention Residuals](https://arxiv.org/abs/2603.15031)：

- **機制**：將 $L$ 層切為 $N$ 個 block（Inkling 為每 6 層一個 block，$v_0 = \mathrm{RMSNorm}(\mathrm{embed}(x))$），block 內各子層輸出累加為區塊和 $v_j$ 與當前部分和。每個讀取點以一個**與輸入無關、初始化為 0、在 FP32 運算**的可學查詢向量 $q \in \mathbb{R}^d$ 做深度加權：

$$x = \sum_j \alpha_j \hat{v}_j, \qquad \alpha_j = \mathrm{softmax}_j\!\left(\frac{q^\top \hat{v}_j}{\sqrt{d}}\right), \qquad \hat{v}_j = \frac{v_j}{\mathrm{RMS}(v_j)}$$

- **I/O 與消融對照**：依 Kimi 論文（436M, 16 層），改用依輸入計算的 query 雖可再降 loss（1.731 vs 1.737），但每層多 $d\times d$ 投影且解碼需循序執行。Block AttnRes 每 token 每層 I/O 僅 $(N/S + 5)d \approx \mathbf{5.5d}$（標準殘差 $3d$、Full AttnRes $24d$、mHC $34d$），訓練開 PP 開銷 <4%、推論採兩階段 online softmax 延遲開銷 <2%，在縮放律上相當於基線多 **1.25×** 算力。
- **Inkling 2 Mini 的 56 個讀取點**：權重包含 `attn_queries.0..22`（其中第 4、10、16、22 組形狀為 `[3, 2304]`，對應 4 個全域層的 Q/K/V **各自獨立查詢**，其餘為 `[2304]`，共 31 個）+ `mlp_queries.0..23`（24 個，讀取區塊和＋注意力輸出）+ `output_query`（1 個）= **56 個讀取點**。全域層 Q/K/V 分開讀取是 Inkling 在原論文基礎上的架構變體。

---

## 四、單層前向資料流與 1M 上下文注意力設計

### 4.1 單層前向計算（依權重形狀與 NeMo 文件重建）

```text
[Attention Sublayer]
  1. AttnRes 深度混合（全域層 Q/K/V 各自獨立混合）
  2. (Core 層第 2、3 輪) Rank-64 閘門混入跨輪訊息 m_s
  3. RMSNorm（滑動層單一 attn_norm；全域層 q/k/v_input_norm 各自 norm）
  4. 線性投影：
       q = wq_du(x)   [3072, 2304]  (24 heads x 128)
       k = wk_dv(x)   [1024, 2304]  (8 KV heads x 128)
       v = wv_dv(x)   [1024, 2304]  (8 KV heads x 128)
       r = wr_du(x)   [384,  2304]  (24 heads x d_rel=16)
  5. K/V 短卷積 (kernel=4)：滑動層靜態 k_sconv/v_sconv；全域層靜態 static_w + 動態 weight_proj
  6. 範數截斷與正規化：
       q 乘可學 q_l2_clip.scale [24, 128] 後與 k 皆截斷 L2 至 sqrt(512) ≈ 22.63
       r 截斷 L2 至 8.0；V 做無 gamma 的 RMSNorm
  7. logits = (q · k) / sqrt(128) + rel_profile(r, Δ) + sink_logit(attn_sink_proj)
  8. softmax（滑動層 window=1024；全域層含 log_scaling）
  9. y = SDPA ⊙ gate_attn(x) [3072, 2304]  -->  wo_ud [2304, 3072]  -->  attn_sconv [2304, 1, 4]
 10. 累加進當前 block 總和

[MoE Sublayer]
  1. AttnRes 深度混合（讀區塊和 + Attention 輸出）
  2. (Core 層第 2、3 輪) Rank-64 閘門混入跨輪訊息 m_moe
  3. mlp_norm --> Router (128 選 8, F32 bias, 正規化 x 8.0)
  4. 8 個 capped SwiGLU 專家加權求和 --> mlp_sconv [2304, 1, 4]
  5. 累加進當前 block 總和

[Output]
  output_query 深度混合 --> final RMSNorm --> FP32 unembed --> 30*tanh(z/30) (id >= 200058 填 -inf)
```

### 4.2 五項關鍵注意力組件與 KV Cache VRAM 估算

1. **5:1 滑動視窗＋GQA（24Q / 8KV）與 KV Cache VRAM 計算**：
   每 6 層為 5 層 SWA（`window=1024`）+ 1 層全域（與 [Gemma 3](https://arxiv.org/abs/2503.19786) 完全一致）。因 48 次層訪問各自保有獨立 KV cache，若全為全域層，BF16 KV 需要 $48 \times 2 \times 8 \times 128 \times 2\text{ B} = 192\text{ KiB/token}$（1M tokens 約 192 GiB）。實際隨長度線性增長的只有 **8 次全域訪問**（第 5 層 1 次＋第 11、17 層各 3 次＋第 23 層 1 次），約 **32 KiB/token**（1M tokens 約 32 GiB）；其餘 40 次滑動視窗訪問每條序列固定約 **160 MiB**，整體 KV 節省約 6 倍。  
   *註：此理論估算與 vLLM recipe 所寫「單張 GB200 約可同時 1.8 個 1M 請求」不完全吻合，框架額外開銷尚未查明。*
2. **以 Learned Relative Position Logits 取代 RoPE**：
   語言骨幹完全不用 RoPE，而是由 `wr_du` `[384, 2304]` 產生每頭 `d_rel=16` 的相對位置特徵 $r_{i,h}$，截斷 L2 至 8.0 後與 `rel_logits_proj.proj` $T \in \mathbb{R}^{16 \times 1024}$ 內積得到距離 $\Delta < 1024$ 的偏置表（vLLM 為此專門實作 sheared-bias FlashAttention-4 kernel）。
3. **依輸入產生的 Zero-Valued Virtual Sink**：
   相較 [StreamingLLM](https://arxiv.org/abs/2309.17453) 保留前 4 個真實 token，或 [gpt-oss](https://arxiv.org/abs/2508.10925) 在 softmax 分母加常數項 $e^{b_h}$，Inkling 以 `attn_sink_proj` `[24, 2304]` 由當前 token 動態算出每頭 sink logit，對應 value 恆為 0 的虛擬 token，只吸收多餘機率質量。
4. **逐頭逐通道輸出閘（Gated Attention）**：
   `gate_attn` `[3072, 2304]` 對應 [Qwen Gated Attention (2025)](https://arxiv.org/abs/2505.06708) 的 **G1 elementwise** 位置（SDPA 後、$W_O$ 前）。在 Qwen 15A2B MoE 消融中，G1 elementwise 可將首 token 注意力由 0.467 壓低至 0.048、最大 activation 由 1,053 壓低至 94、128K 外推 RULER 由 31.65 提升至 58.82，延遲開銷 <2%。
5. **每層 4 個短卷積（`kernel=4`）與全域層動態 K/V 卷積**：
   - 20 個滑動層的 K/V 採用靜態 depthwise causal conv1d（`[1024, 1, 4]`，近似 [Canon-B](https://arxiv.org/abs/2512.17351)）。
   - 4 個全域層的 K/V 結合 `static_w` `[4, 1024]` 與 `weight_proj` `[32, 2304]`（由 token 動態生成 8 KV heads × 4 taps 權重，屬 [Dynamic Short Convolutions](https://arxiv.org/abs/2606.03825) 的 head-wise 參數化）。
   - 全部 24 層在注意力輸出與 MoE 輸出後各有一個 `attn_sconv` / `mlp_sconv` `[2304, 1, 4]`。

---

## 五、MoE 剪枝、1T 蒸餾修復與 On-Policy 後訓練

### 5.1 MoE 實作細節

- **路由與負載平衡**：128 個路由專家、top-8、`n_shared_experts=0`、專家中間寬度 1,152。Router 含 BF16 `gate.weight` `[128, 2304]` 與 F32 `gate.bias` `[128]`（NeMo 強制 F32 計算，對選中分數正規化後乘 `route_scale=8.0`）。
- **Capped SwiGLU**：`mlp_gate_softcap=4.0`、`mlp_up_softcap=25.0`，推論形式為 $4\tanh(g/4)\cdot \sigma(g)\cdot 25\tanh(u/25)$（sigmoid 吃未截斷的 $g$），routing 權重在 down 投影前乘上。

### 5.2 為何能在中期剪掉 2/3 專家（384 → 128）

將 MoE 壓縮進 ~20 GB 預算有「合併專家」與「剪枝專家」兩條路。[REAP (2025)](https://arxiv.org/abs/2510.13999) 從數學上證明合併專家會導致 **functional subspace collapse**：設 router 對兩專家的混合比例 $r(x) = g_i / (g_i + g_j)$，合併成 $\tilde{f} = \alpha f_i + (1-\alpha) f_j$ 的最小誤差為：

$$\mathbb{E}\big[(g_i+g_j)^2 (r-\alpha^\star)^2 \lVert f_i - f_j\rVert^2\big] \approx \mathbb{E}[(g_i+g_j)^2]\cdot \mathrm{Var}[r]\cdot G_{ij}$$

只要 $\mathrm{Var}[r] > 0$ 誤差即無法消除，實證中合併後專家輸出分佈範圍最多縮小 100×。但另一方面，REAP 也證實**不重訓的 one-shot 頻率剪枝**在 50% 壓縮率下會直接崩潰（Qwen3-Coder-480B 程式分數 0.636 → 0.010、Kimi-K2 0.631 → 0.056）。

Inkling 2 Mini 採用頻率準則一次剪掉 **66.7%**（384 → 128），其成立前提在於剪枝發生在訓練中途，並緊接著以更大的內部 Inkling 老師模型執行 **1T tokens 蒸餾修復（healing）**——這是 NVIDIA [Minitron](https://arxiv.org/abs/2407.14679) 修復規模（94B tokens）的約 10 倍，足以讓剩餘 128 個專家重新吸收被刪專家的表徵功能。

### 5.3 後訓練管線：SFT → 2.2T RL → On-Policy 蒸餾對齊

- **RL 階段**：2,000 步、約 2.2T tokens（平均每步 1.1B tokens），負責拉高程式與長程代理能力。
- **On-Policy 蒸餾對齊**：源於 Google DeepMind [GKD (ICLR 2024)](https://arxiv.org/abs/2306.13649)。TML 版本固定 $\lambda = 1$ 完全由學生採樣軌跡、`discount = 0`，老師僅對學生生成的 token 做一次前向算 logprob，直接把 RL advantage 設為負的逐 token reverse KL：

$$\mathrm{advantage}_t = -\Big(\log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{\text{teacher}}(y_t \mid x, y_{<t})\Big)$$

RL 每個 episode 僅提供 $O(1)$ bits 稀疏獎勵，逐 token 蒸餾則提供 $O(N)$ bits 密集回饋，且可直接沿用既有 RL importance-sampling loss 基礎設施（TML 引 Qwen3 對照：AIME'24 跑 RL 需 17,920 GPU-hr 得 67.6%，On-policy 蒸餾僅 1,800 GPU-hr 得 74.4%）。

---

## 六、端到端 NVFP4 訓練與 Four Over Six 量化

### 6.1 低精度格式與 TML vs NVIDIA 配方對照

| 格式 | 元素 | 區塊 | 縮放因子 | 每值位元 | GB300 密集算力 |
| --- | --- | --- | --- | --- | --- |
| **NVFP4** | E2M1（±6） | 16 | FP8 E4M3 + 每張量 FP32 | 4.5 | 15 PFLOPS |
| **MXFP4** | E2M1 | 32 | E8M0（僅 2 的冪） | 4.25 | — |
| **MXFP8** | E4M3 / E5M2 | 32 | E8M0 | 8.25 | 5 PFLOPS |
| **BF16** | — | — | — | 16 | 2.5 PFLOPS |

TML 在 [End-to-End 4-Bit Training](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/)（實驗對象為 Inkling 2 Small，配方沿用至 Mini）與 NVIDIA [12B NVFP4 預訓練論文](https://arxiv.org/abs/2509.25149) 的關鍵差異：

| 項目 | NVIDIA 12B 配方 | TML 配方（Inkling 2） |
| --- | --- | --- |
| **高精度層** | 前 2 + 後 8 block，BF16（~16%） | 最後 1/6 層，**MXFP8**（Mini 24×1/6 = 第 20–23 層） |
| **權重縮放** | 16×16 二維縮放 | **1×16**（認為 16×16 太粗）＋每專家外層 FP32 縮放 |
| **激活縮放** | 1×16 | 1×16 + 每 token 外層縮放（保持 batch invariance、防因果洩漏）+ **Four Over Six** |
| **反向傳遞** | NVFP4 + 隨機捨入 (SR) | **MXFP8 + SR**（只量化前向） |
| **Hadamard (RHT)** | 16×16 用在 Wgrad 輸入 | 32 維 RHT + S2F6 用在 **MoE combine 通訊**（16 → 8.25 bit） |
| **末段切回 / Kernel** | 最後 18% tokens 切回 BF16 | 不切回；完全確定性 BBBK kernel，**部署即訓練精度** |

在 GB300 NVL72 上，TML 配方預訓練比全 MXFP8 快 **8%**、RL 快 **20%**、算力效率損失約 2%，且無訓練與推論的數值落差。

### 6.2 Four Over Six（4/6）激活量化原理

NVFP4 的 E2M1 僅能表示 $\pm\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$。若固定將 16 元素區塊的最大值縮放到 6，由於 4 與 6 之間沒有可表示點，落在最大值 66.6%–100%（約 5 附近）的數值誤差最大。[Four Over Six (2025)](https://arxiv.org/abs/2512.02010) 對每個區塊同時試算縮放到 $M=4$ 與 $M=6$，擇誤差較小者：

$$\Delta^{(M)} = \mathrm{fp8}\!\left(\frac{\max|X|}{M}\right), \qquad E^{(M)} = \frac{1}{n}\sum_i\big(\bar{X}^{(M)}_i\Delta^{(M)} - X_i\big)^2, \qquad M \in \{4, 6\}$$

例如 `[10, 20, 30, 40]` 縮放到 6 得 `[1.5, 3, 4, 6]`（MSE 4.33），縮放到 4 得 `[1, 2, 3, 4]`（MSE 0）。實作須將 tensor scale 上限由 448 降至 256（避免選 4 時 scale 達 672 溢出 E4M3）；在 Blackwell 上以 PTX `cvt` 留在暫存器，量化 kernel 額外開銷 <15%。論文消融顯示**僅對激活值套用 4/6 即拿回幾乎全部收益**（且 MXFP4 因 E8M0 無法表示 1.5× scale 而不適用）。

---

## 七、DSpark 推測解碼與多框架 VRAM / 吞吐實測

### 7.1 DSpark Drafter 結構

倉庫附帶的 `mtp.safetensors`（636 MB）為 3 層 [DSpark (2026-07)](https://arxiv.org/abs/2607.05147) 半自迴歸 drafter（全模型唯一使用 RoPE 的模組，`rope_theta=1e6`，第 0、2 層為 SWA）：

- **跨迴圈特徵融合**：讀取目標模型 `target_layer_ids=[5, 23, 41, 48]`——依序對應 Prelude 末層（5）、Core 第 2 輪第 11 層（23）、Core 第 3 輪第 17 層（41）三個全域層訪問點與最終狀態（48），經 `fuse_weight` `[4, 4]` 融合後以 `k_proj_ctx`/`v_proj_ctx` 注入 drafter 的 K/V。
- **半自迴歸與信心排程**：平行 backbone 一次產出 `block_size=8` 個基礎 logits $U_k$ 與隱狀態 $h_k$，再由 rank-256 的 GRU 式 `markov_head`（`markov_w1` `[201024, 256]`、`rnn_proj` `[768, 5120]`、`hidden_out` `[2304, 256]`）補回 block 內前綴依賴；Confidence head 預測前綴存活機率 $a_{r,j} = \prod_{i \le j} c_{r,i}$，全域貪婪排程最大化 $\Theta = \tau \cdot \mathrm{SPS}(B)$。

### 7.2 主流框架部署限制與實測速度

| 框架 | 關鍵設定與 VRAM / 並行限制 | 實測數據 |
| --- | --- | --- |
| **vLLM** (`>=0.35.0`, cu130, r580+) | `--reasoning-parser inkling`；DSpark 設 `num_speculative_tokens=7`；單卡 TP1 可跑 1M，H100 / DGX Spark 上限 262,144 | 每用戶 100 tok/s 下：32K 並發 ~45、64K ~32、128K ~20 |
| **SGLang** | **NVFP4 專家不可 TP 切分，因此 EP 必須等於 TP**（GB300/H200/MI35x 用 TP4/EP4；RTX/Spark TP1）；`--speculative-algorithm DSPARK`；`--inkling2-loop-share-cache` 可讓三輪共用 KV（KV pool ~2×） | RTX 5090（32 GB VRAM）上下文上限 **65,536**（開 DSpark 降為 **32,768**）；gsm8k：1 輪 84.0% / 3 輪 88.5% / 3 輪+share-cache 90.0% |
| **llama.cpp (Unsloth)** | `--spec-type draft-dspark --spec-draft-n-max 2`（drafter Q8_0 0.4 GB）；`mmproj-F16` 639 MB；KV Q8_0 慢 3–7% 但上下文約加倍 | Q4_K_XL 解碼：RTX PRO 6000 **125.2**、B200 **110.8**、A100-40G **67.0**、L4 **42.9**、Ryzen AI Max **32.8**、T4 **23.9** tok/s；B200 開 DSpark greedy 99.9 → **117.3** tok/s |

Unsloth GGUF 量化品質對照（推薦 **UD-Q4_K_XL**）：

| 量化檔 | 大小 | KLD | Top-1 一致率 |
| --- | --- | --- | --- |
| `UD-IQ1_S` | 7.76 GB | 0.0911 | 89.4% |
| `UD-IQ2_XXS` | 8.49 GB | 0.0726 | 90.6% |
| `UD-Q3_K_XL` | 13.63 GB | 0.0174 | 96.3% |
| `UD-Q4_K_M` | 16.61 GB | 0.0108 | 97.9% |
| **`UD-Q4_K_XL`** | **17.63 GB** | **0.0107** | **98.0%** |
| `UD-Q8_K_XL` | 22.87 GB | — | — |

---

## 八、可解釋性（Goodfire）與 CoT 可監控性

### 8.1 Goodfire 權重空間與資訊流分析

Goodfire 掃描 3.5M 個 writer × 104 個 reader（約 3.68 億對，$|\cos| > 0.30$ 有 3,490 對），觀察到三個明確現象：

1. **Writer 集中在 Block 末層**：深度方向的 writer 集中在第 6、12、18 層（1 起算）；跨迴圈 writer 集中在區塊 3，尤其第 18 層寫入第 13 層，為激活打標籤以供 AttnRes 讀取。區塊 3 幾乎總是讀取自己上一輪的輸出。
2. **離群通道 $d_{2027}$ 的跨輪反轉**：RMS 正規化後平均 +32.4，佔平方範數 46%。第 18 層在第 1、2 輪呈正向尖峰，第 3 輪反轉為負向尖峰，扮演最後一輪的收尾清理。
3. **Logit-lens 失效與 Coda-lens 有效**：傳統 logit-lens 只在每輪末層有效，但 tuned-lens 與 coda-lens 效果良好（1,250 份 FineWeb 文件），顯示中間迴圈確實在非詞表的潛空間內演算。

### 8.2 CoT 可監控性檢查

由於部分推理移入不可見的迴圈隱狀態，TML 執行了兩項測試：
- **[CoT-Control-QA](https://arxiv.org/abs/2603.05706)**（CoT 750–1,250 tokens）：遵循率最多 **0.1%**（Inkling 2 Small 0.3%、Qwen3.8-27B 1.3%）。
- **LinuxArena 隱蔽任務**（Qwen3.6-35B-A3B 監控器、1% 誤報率）：只看工具呼叫成功率 3.9%，含思考內容降至 **1.3%**（Qwen3.8-27B 為 20.5% / 0.4%；varloop 1–5 輪含思考為 0.3% / 0.4% / 0.6% / 0.3% / 0.0%，無隨迴圈惡化趨勢）。

---

## 九、評測基準、硬體效率與尚未公開的缺口

### 9.1 官方評測（節選）與 Artificial Analysis 獨立實測

| 類別 | 評測 | Inkling 2 Mini<br/>(5.4B/26.4B) | Qwen3.8-27B<br/>(27B Dense) | Gemma 4 31B<br/>(30.7B Dense) | Nemotron 3.5 Super VL<br/>(12B/124.5B) |
| --- | --- | --- | --- | --- | --- |
| **綜合** | AA Intelligence Index v5.0 | **20.1** | **20.1** | 4.4 | 13.6 |
| **程式** | DeepSWE v1.1 (%) | **48.2** | 42.2 | 0.3 | 24.7 |
| **程式** | SWE-bench Verified (%) | **79.2** | 77.8 | 46.1 | 71.1 |
| **程式** | SWE-bench Pro v2 Hard (%) | **77.9** | 73.8 | 14.2 | 44.1 |
| **程式** | Terminal Bench v2.1 (%) | 71.6 | **73.0** | 43.4 | 60.7 |
| **程式** | Vibe Code Bench v1.1 (%) | **77.9** | 64.9 | – | 3.5 |
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

- **Artificial Analysis 獨立評測**：Intelligence Index 20.1 與 Qwen3.8-27B (xhigh) 打平。Inkling 2 Mini 每題平均輸出 **152K tokens**（比 Qwen3.8-27B 的 110K 多 38%），但因活躍參數少約 5 倍，「活躍參數 × tokens」總計算量仍少 **2.8×**；每題成本 **$0.59 / 9.0 分鐘**（Qwen3.8-27B 為 $3.53 / 38.2 分鐘）。
- **Intelligence per Watt（Stanford, DGX Spark）**：Inkling 2 Mini（MXFP8, low effort）**0.703** vs Qwen3.6-35B-A3B（NVFP4）0.430 vs Gemma-4-26B-A4B（NVFP4）0.159。

### 9.2 實務限制與尚未公開／不一致的細節

- **官方自述限制**：事實知識回憶落後同級稠密模型且易自信說錯；多輪對話（MultiChallenge 50.0%）與複雜指令（ComplexConstraints 33.4%）較弱；除錯時可能陷入長推理迴圈（即使 `effort=none` 也無法完全關閉），實務必須設 token 與回合上限；Agent 有未確認即刪改檔案的紀錄；中文輸出會出現措辭簡略與中英混雜。
- **未公開資訊**：預訓練總 token 數、資料配比、知識截止日、1M 長上下文延展策略、RL 獎勵設計與總訓練算力。
- **尚未從原始碼驗證的公式與口徑差異**：`log_scaling_alpha=0.1` 的確切縮放式、`gate_attn` 激活函數、Router 分數函數（sigmoid vs softmax）、sink 進入 softmax 的確切實作，以及研究版與 Mini 在 `ARC-AGI-2`（49.3 vs 51.3）與 `MRCR 512K-1M`（81.4 vs 37.7）上的口徑差異。後續可直接檢視 SGLang PR #51926、llama.cpp PR #30817 或 `transformers` 的 `inkling2_loop` 實作補齊。

---

## 參考資料

- [Introducing Inkling 2 Mini（Thinking Machines Lab）](https://thinkingmachines.ai/news/inkling-2-mini/) / [TML Model Card](https://thinkingmachines.ai/model-card/inkling-2-mini/) / [TML-d24-varloop-research](https://thinkingmachines.ai/model-card/tml-d24-varloop-research/)
- [Hugging Face Repo & config.json](https://huggingface.co/thinkingmachines/Inkling-2-Mini) / [Welcome Inkling 2 Mini（HF Blog）](https://huggingface.co/blog/inkling2mini)
- [TML: End-to-End 4-Bit Training](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/) / [TML: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Goodfire: Interpreting Inkling 2 Mini](https://www.goodfire.com/research/interpreting-inkling-2-mini/)
- [Artificial Analysis: Inkling 2 Mini Evaluation](https://artificialanalysis.ai/articles/inkling-2-mini)
- [vLLM Recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini) / [SGLang Cookbook](https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini) / [Unsloth Docs](https://unsloth.ai/docs/models/inkling-2-mini)
- 相關論文：[Universal Transformers (1807.03819)](https://arxiv.org/abs/1807.03819)、[Huginn (2502.05171)](https://arxiv.org/abs/2502.05171)、[Reasoning with Latent Thoughts (2502.17416)](https://arxiv.org/abs/2502.17416)、[Ouro / LoopLM (2510.25741)](https://arxiv.org/abs/2510.25741)、[Attention Residuals (2603.15031)](https://arxiv.org/abs/2603.15031)、[Gated Attention (2505.06708)](https://arxiv.org/abs/2505.06708)、[Dynamic Short Convolutions (2606.03825)](https://arxiv.org/abs/2606.03825)、[REAP (2510.13999)](https://arxiv.org/abs/2510.13999)、[Minitron (2407.14679)](https://arxiv.org/abs/2407.14679)、[GKD (2306.13649)](https://arxiv.org/abs/2306.13649)、[NVFP4 Pretraining (2509.25149)](https://arxiv.org/abs/2509.25149)、[Four Over Six (2512.02010)](https://arxiv.org/abs/2512.02010)、[DSpark (2607.05147)](https://arxiv.org/abs/2607.05147)、[Think Fast (2606.07157)](https://arxiv.org/abs/2606.07157)、[CoT-Control (2603.05706)](https://arxiv.org/abs/2603.05706)、[Intelligence per Watt (2511.07885)](https://arxiv.org/abs/2511.07885)
