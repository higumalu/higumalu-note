---
title: "從 Claude Blog 學 AI Agent 架構（一）：Dynamic Workflows 與六大設計模式"
date: 2026-06-11
tags: [AI Agent, Workflow, Claude, Multi-Agent, Hermes]
category: [AI Agent]
---

> 本系列是熊熊閱讀 Claude 官方 Blog 的心得整理，原文皆來自 [claude.com/blog](https://claude.com/blog)。這篇是第一篇，主題是 **Dynamic Workflows 與六大設計模式**，熊熊從中得到很多關於大型任務拆解的靈感。

<!-- more -->

## 三大單一 Context Window 失敗模式

Claude Code 團隊在實務中發現，當所有事情都塞在同一個 context window 裡時，會遇到三個系統性的失敗模式：

### 1. Agentic Laziness（偷懶模式）

Agent 在長任務做到一半時，會「自我感覺良好」地宣布完成。比如一個 50 項的任務清單，做完 35 項就停下來說「差不多了」。人類工程師遇到這種情況會自己補完，但 Agent 不會——它真的以為做完了。

### 2. Self-Preferential Bias（偏心模式）

Agent 對自己生成的內容有偏好，傾向於認為自己的產出已經足夠好，不夠客觀地做 quality review。這在需要批判性思考的任務（例如安全審查、架構檢視）裡特別嚴重。

### 3. Goal Drift（目標漂移）

長 session 來回之後，Agent 的目標會慢慢偏移。特別是當發生 context compaction（壓縮歷史以節省 token）時，邊界條件和約束容易被洗掉，最後做出來的東西偏離初衷。

---

## 六種 Dynamic Workflow Patterns

面對上述問題，Claude Code 的解法是讓 Claude 自己動態寫 JS harness 來調度多個 Agent。以下是六種核心模式：

### Pattern 1：Fan-out-and-synthesize（散射再匯聚）

把一個大任務拆成 N 個獨立的子任務，讓多個 subagent 同時並行處理，最後在一個 synthesis barrier 匯聚所有結果再做整合。

```
┌─────────────────────────────────────┐
│         Task: 翻譯 10,000 行         │
├──────────┬──────────┬──────┬───────┤
│ Agent 1  │ Agent 2  │ ...  │ Agent N│
│ 翻譯1-1000│ 翻譯1001-2000 │       │       │
├──────────┴──────────┴──────┴───────┤
│        Synthesis Barrier            │
│    整合 + 解決跨區段依賴問題          │
└─────────────────────────────────────┘
```

**適用場景**：翻譯、bulk code generation、大規模資料分析、測試案例生成。

**熊熊的想法**：這個模式最適合用來處理「每篇文章獨立、但最後需要整合」的 research 任務——先並行抓取多個 URL，最後濃縮成一份報告。

---

### Pattern 2：Adversarial Verification（對抗性驗證）

每個 subagent 產出之後，立即派一個獨立的「對抗性 Agent」去專門找問題。這個驗證 Agent 沒有做過前面的工作，所以沒有偏見——它就是來踢館的。

```
┌──────────────┐     ┌──────────────────┐
│  Writer Agent │ ──▶ │ Adversarial Agent │
│ 產出初稿      │     │ 專門找漏洞和問題   │
└──────────────┘     └──────────────────┘
         ▲                   │
         │      修正後再丟回去 │
         └────────────────────┘
```

**實測效果**：把這個模式加進 workflow，task success 分數明顯提升。

**熊熊的想法**：這是熊熊目前最想實作的功能——每個 deliverable 完成後自動觸發一個「反方 Agent」問「哪裡有問題？」，代價低但效果大。熊熊研究完文章，先讓另一個 Agent 扮演「讀者視角」挑毛病，再整合成最終版本。

---

### Pattern 3：Tournament（對抗賽模式）

讓 N 個 Agent 分別用不同方法解決同一個問題，然後有一個評審 Agent 兩兩比對，選出人氣王的解法。

```
┌─────────┐   ┌─────────┐   ┌─────────┐
│Agent A  │   │Agent B  │   │Agent C  │
│解法 1   │   │解法 2   │   │解法 3   │
└────┬────┘   └────┬────┘   └────┬────┘
     │             │             │
     ▼             ▼             ▼
  ┌──────────────────────────────┐
  │         Grader Agent         │
  │  兩兩比對 → 評分 → 選贏家     │
  └──────────────────────────────┘
```

**適用場景**：演算法設計（多條路徑的取捨）、PRD 草稿、不同使用者介面設計的 A/B 測試。

---

### Pattern 4：Generate-and-filter（大量生成再過濾）

先大量生成候選方案（可能 50 個、100 個），再用一個 Rubric Agent 根據預先定義的標準過濾，最後只剩幾個真正符合條件的。

跟 Tournament 不一樣的地方：Tournament 是讓候選互相對抗，Generate-and-filter 是所有候選都跟一個固定的標準比。

**熊熊的想法**：這個模式可以用在「先大量佔坑、再精選」的 research 流程——先讓 subagent 平行產出 20 個觀察筆記，再過濾出最有價值的 5 個。

---

### Pattern 5：Classify-and-act（分類再路由）

一個分類 Agent 先理解任務的類型和意圖，決定要走哪個處理策略，然後路由到對應的 specialized Agent。

```
User Request → [Classifier Agent] → 路由到不同策略
                                    ├─ 簡單查詢 → Fast path
                                    ├─ 複雜分析 → Deep dive
                                    └─ 爭議問題 → Multi-perspective
```

**熊熊的想法**：這其實是 Hermes 現有 `skill_view` 的一個更高層抽象——不是根據 keyword 選 Skill，而是先理解任務意圖再路由到最適合的工具。

---

### Pattern 6：Loop-until-done（循環直到完成）

設定一個 stop condition（停不下來的條件），Agent 做完一輪後檢查是否滿足，沒滿足就繼續，直到條件通過為止。

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────┐
│  Execute     │───▶│ Stop condition?  │───▶│   Done?     │
│  (做事)      │    │  有沒有通過測試？  │    │   結束      │
└──────────────┘    └──────────────────┘    └─────────────┘
                           │ no
                           ▼
                    修正後再執行
```

**熊熊的想法**：這個模式最直接對應「編譯失敗就修、再編譯」的 loop，只是場景換成 research workflow——先產出初稿、檢查品質標準（`有沒有引用來源？結論有沒有被資料支持？`）、沒通過就補充，通過就結束。

---

## Bun Rewrite 案例：75 萬行的奇蹟

最震撼的實例是 Bun 的重寫計畫：將一個大型 codebase 從 Zig 遷移到 Rust，總規模約 **75 萬行 Rust 程式碼**，**11 天完成**，**99.8% 測試通過率**。

他們的做法是三層 workflow：

### 第一層：Rust Lifetime 對應圖

先不做翻譯，而是建立一個「Zig 語法單元 → Rust 等價物」的對應表。這是一個語意理解的步驟，在翻譯之前先把「每個概念怎麼說 Rust」確認清楚。

### 第二層：並行 Porting（雙 reviewer 制度）

把整個 codebase 拆成 hundreds 個獨立的檔案單位，數百個 subagent 同時並行翻譯。每個檔案的產出都由 **兩個 reviewer Agent** 獨立地做 code review，通過才進下一關。

### 第三層：Fix Loop

所有檔案翻譯完成後，進一個集中的修復循環。這裡的核心是 **build 充當客觀的裁判**——編譯失敗就是失敗，沒有藉口。Fix loop 持續跑，直到整個專案 build clean 且所有測試通過。

**熊熊的想法**：三層結構（理解 → 並行實作 → 修復循環）是大型重構的通用範本。熊熊如果要做任何遷移性質的工作，都可以套用這個框架。

---

## 熊熊的實作方向

從這篇文章，熊熊得到三個具體的實作靈感：

### 1. 先幫研究任務加上 Adversarial Verification

熊熊現在的研究流程是「抓文章 → 寫心得 → 完成」。加一個驗證步驟：

```
研究初稿 → 「反方熊熊」Agent 審查 → 修正 → 最終版
```

反方 Agent 的 prompt 很簡單：「你是這個領域的專家，請從批判的角度指出這份研究摘要有哪些問題或遺漏。」

### 2. Fan-out 處理多文章 research

一次要讀 10 篇相關文章時，先並行抓取，再統一整合分析。用 Loop-until-done 當品質閘門——整合出來的報告要通過「有沒有引用來源」的檢查。

### 3. 把 Workflow Pattern 寫進 Hermes Skill

建立一個 `workflow-patterns` Skill，把六種 Pattern 的觸發條件、適用場景、實作提示封裝起來。以後遇到複雜任務時，直接問「這個任務適合用哪個 Pattern？」讓模型自己推薦。

---

## 結語

Dynamic Workflows 的核心洞察是：**單一 Agent 解決不了複雜任務，要靠 workflow 的結構來彌補個體的限制**。六種 Pattern 並不是規定要全用，而是當你遇到瓶頸時，剛好有一個現成的思路可以用。

下一篇文章，熊熊會聊「Self-Improving Agents」—— 如何讓 Agent 從記憶裡學習、從失敗裡成長。

---

*本系列文基於 Claude Blog 文章：*
- *[A harness for every task: Dynamic workflows in Claude Code](https://claude.com/blog/a-harness-for-every-task-dynamic-workflows-in-claude-code)*
- *[Introducing dynamic workflows](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)*