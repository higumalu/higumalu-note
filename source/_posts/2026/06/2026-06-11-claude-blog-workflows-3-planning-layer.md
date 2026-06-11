---
title: "從 Claude Blog 學 AI Agent 架構（三）：Planning Layer 與大型重構範本"
date: 2026-06-11
tags: [AI Agent, Planning, CodeRabbit, Multi-Agent, Hermes]
category: [AI Agent]
---

> 本系列是熊熊閱讀 Claude 官方 Blog 的心得整理，原文皆來自 [claude.com/blog](https://claude.com/blog)。這篇是第三篇，主題是 **Planning Layer** 與大型重構實戰——包括 CodeRabbit 的 Planner 系統，以及那個震撼人心的 Bun 案例。

<!-- more -->

## 為什麼需要 Planning Layer

大多數 AI coding 工具的流程是這樣的：

```
User: 幫我做一個登入功能
AI:   好的，開始寫 code
      寫好了！測試通過！
```

看起來很完美，但 CodeRabbit 發現了一個深層的問題：**程式碼通過編譯和測試，但做的不是團隊真正想做的事**。

原因是：**開發者的假設（implicit assumptions）AI 並不知道**。比如「我們的登入頁面有兩步驟驗證」「密碼重置流程要走 Slack webhook」「登入失敗要記錄到 Datadog」。這些東西從來沒有寫在任何規格文件裡，AI 當然不知道。

所以 CodeRabbit 在 Implementation 之前，多加了一層 **Planning Layer**。

---

## CodeRabbit 的 Planner 系統

### 核心架構

```
User Request
       ↓
┌──────────────────┐
│  PRD Planning    │  ← Opus 模型
│  (規劃階段)       │
└────────┬─────────┘
         ↓ 產出結構化 PRD
┌──────────────────┐
│  PRD Review      │  ← 人類審核
└────────┬─────────┘
         ↓ 人類確認
┌──────────────────┐
│  Implementation  │  ← Sonnet 模型
│  (實作階段)       │
└──────────────────┘
```

### 三階段各有分工

**第一階段：PRD Planning（Opus）**

這個階段用最聰明的模型（Opus）來做複雜的協調工作：
- 分析 user request 的意圖和邊界
- 理解 code base 的結構和限制
- 補足隱性假設，把它們變成顯性的規劃文件
- 預估實作風險和依賴關係

**第二階段：Human Review（熊熊 / 人類）**

規劃文件出來之後，人類審核確認。有問題就修改規劃，沒問題才往下走。這一步的核心價值是：**在浪費大量時間之前，先對焦「要做什麼」**。

**第三階段：Implementation（Sonnet）**

有了清晰的 PRD，實作階段的模型可以降級到 Sonnet——因為不確定性已經大幅降低了。Sonnet 擅長把結構化的步驟轉化成乾淨的程式碼。

### 熊熊的想法

Planning Layer 的精神其實就是「謀定而後動」。熊熊的大型研究任務如果也能先跑一個 planning phase——讓模型先分析「這次研究要覆蓋哪些維度」「哪些文章要先讀」「結論要怎麼組織」——就不會做到一半發現方向偏了，還要回頭重來。

---

## Model Routing：讓對的模型做對的事

CodeRabbit 的另一個關鍵設計是 **Model Routing**——不同複雜度的任務，分配給不同能力的模型：

| 模型 | 角色 | 擅長什麼 |
|---|---|---|
| **Opus** | 協調者 / Planner | 複雜推理、架構設計、風險評估 |
| **Sonnet** | 實作者 | 結構化步驟轉化成乾淨程式碼 |
| **Haiku** | 窄範圍操作員 | 簡單的 distilling、格式化、單一工具呼叫 |

這個路由背後的原則是：**昂貴的模型只用在真正需要的地方**。Haiku 的成本是 Opus 的 1/50，如果一個任務只需要簡單的字串處理，強迫用 Opus 就是浪費。

### 熊熊的想法

熊熊的 research workflow 其實已經隱含了 routing：

- 複雜的「文章理解與架構規劃」→ 用主力模型（主力對話）
- 簡單的「並行抓取多個 URL」→ 可以用一個輕量的 subagent
- 最簡單的「純下載」→ 甚至不需要 LLM，直接 curl

把這個 routing 顯式化，未來熊熊就可以對不同的 sub-task 選擇性啟用不同能力的模型，降低成本、提升速度。

---

## Bun Rewrite：三層 Workflow 的巔峰示範

Bun 從 Zig 遷移到 Rust 的案例，是整個系列最有視覺衝擊力的實例：

- **75 萬行** Rust 程式碼
- **11 天**完成
- **99.8% 測試通過率**

這個數字厲害的地方在於：不是 99%，是 99.8%。在 75 萬行的規模下，要同時做到速度快（11 天）和精準度高（99.8%），必須靠 workflow 結構取勝。

### 第一層：語意對應（先做）

在開始翻譯任何一行 code 之前，先建立一個完整的「Zig 概念 → Rust 等價物」對應表。這個步驟看起來慢，但其實是最關鍵的：

- 確認 ownership 和 lifetime 的對應方式
- 確認錯誤處理模式（Zig 的 `try`/`error` → Rust 的 `Result`）
- 確認並發模型的差異（Zig 的 async → Rust 的 tokio）

這個步驟由一個 Planner Agent 執行，沒有其他 subagent 並行——**這是唯一一個需要順序執行的步驟**。

### 第二層：並行翻譯 + 雙 Reviewer

把整個 codebase 拆成 hundreds 個檔案單元，數百個 subagent 同時並行翻譯。每個檔案的產出都由 **兩個獨立的 reviewer Agent** 各自做 code review，通過才進下一關。

雙 reviewer 的設計是關鍵：因為翻譯錯誤很容易發生，而同一個錯誤如果由兩個 reviewer 各自檢查到，偽陽性的機率就大幅降低。

### 第三層：Fix Loop 直到乾淨

所有檔案翻譯完成後，進一個集中的修復循環。這裡的裁判不是人，是 **build system**：

- 編譯失敗 → 修
- 測試失敗 → 修
- 通過了才結束

沒有「我覺得差不多了」的空間，build 就是客觀標準。

---

## 從 Bun 案例學到的 Workflow 設計原則

### 1. 理解在先，翻譯在後

千萬不要跳過「先理解」的步驟。Bun 案例裡，「先做語意對應」這個決策節省了後面大量的重工。如果一開始沒有確認 Zig 和 Rust 的概念對應，翻譯到一半發現 semantics 不匹配，要回頭改的成本極高。

### 2. 把 quality gate 放進 workflow，而不是最後才驗

Bun 案例的雙 reviewer 制度把 quality gate 前移到翻譯當下，而不是翻譯完才發現有問題。Outcomes Pattern 的精神也一樣——讓驗證 Agent 跟著產出 Agent 一起跑，而不是最後再做。

### 3. 讓客觀標準當裁判

讓 build system 當裁判，解決了「誰說什麼才算過」的問題。研究 workflow 裡，「熊熊說通過才算過」是一個主觀標準；如果能設計一個 rubric（引用完整性、邏輯一致性、結論有被資料支持），Grader Agent 的判定就是客觀的，不需要熊熊仲裁每個細節。

### 4. Isolated parallelism 是關鍵

Bun 案例的 subagent 之間幾乎沒有通訊需求——每個檔案的翻譯是獨立的，所以可以無限制地並行。如果任務之間有很強的依賴關係，就要在 workflow 設計上先做「去耦合」——把有依賴的部分先做，剩下的才可以並行。

---

## 熊熊的實作藍圖

### 小型 Workflow：熊熊研究一篇部落格文章

```
┌──────────────────────────────────┐
│  1. Planning: 這篇文章要讀什麼重點 │
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│  2. Fan-out: 核心段落並行濃縮      │
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│  3. Outcomes: Grader 驗證品質     │
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│  4. 熊熊確認 → 寫入心得檔案        │
└──────────────────────────────────┘
```

### 中型 Workflow：熊熊研究一個主題（多篇文章）

```
┌────────────────────────────────────────┐
│  1. Planning: 定義 research dimensions │
├──────────┬──────────┬────────┬─────────┤
│  crawl   │  crawl   │  crawl │  crawl  │
│  文章A   │  文章B   │  文章C │  文章D  │
├──────────┴──────────┴────────┴─────────┤
│  2. Adversarial: 每篇心得找反方審查    │
├──────────┬──────────┬────────┬─────────┤
│  心得A   │  心得B   │  心得C │  心得D  │
├──────────┴──────────┴────────┴─────────┤
│  3. Synthesis: 整合成最終報告          │
└────────────────────────────────────────┘
```

### 大型 Workflow：熊熊的系統級遷移或重構

```
┌──────────────────────────────────────┐
│  Phase 1: Planner（先做）              │
│  理解現狀 → 定義對應規則 → 風險評估    │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│  Phase 2: Isolated Fan-out            │
│  數百個 subagent 並行實作 + 雙 review  │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│  Phase 3: Fix Loop                    │
│  直到 build clean + 測試全過           │
└──────────────────────────────────────┘
```

---

## 結語

三篇文章打下來，熊熊最大的心得是：**Workflow 不是工具，是結構**。工具幫你做事，結構確保你用對的方式做事。當任務變得複雜時，光有好工具不夠——你需要設計讓正確的事情自然發生的結構。

Dynamic Workflows 的六種 Pattern 是起點，Self-Improving 的四層系統是骨架，Bun 的三層 Workflow 是終極示範。三者加起來，就是一個從設計到執行到成長的完整框架。

熊熊接下來會把這些想法慢慢落到 Hermes 的實作上。如果熊熊在過程中有新的心得，會再寫成文章跟大家分享。

---

## 系列文章索引

- [（一）Dynamic Workflows 與六大設計模式](/2026/06/11/claude-blog-workflows-1-dynamic-patterns/)
- [（二）Self-Improving Agents 的四層成長系統](/2026/06/11/claude-blog-workflows-2-self-improving-agents/)
- [（三）Planning Layer 與大型重構範本](/2026/06/11/claude-blog-workflows-3-planning-layer/)（本文）

---

*本系列文基於 Claude Blog 文章：*
- *[How CodeRabbit used Claude to build an agent orchestration system](https://claude.com/blog/how-coderabbit-used-claude-to-build-an-agent-orchestration-system)*
- *[Introducing dynamic workflows](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)*
- *[A harness for every task: Dynamic workflows in Claude Code](https://claude.com/blog/a-harness-for-every-task-dynamic-workflows-in-claude-code)*