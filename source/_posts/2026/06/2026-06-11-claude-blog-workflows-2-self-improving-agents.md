---
title: "從 Claude Blog 學 AI Agent 架構（二）：Self-Improving Agents 的四層成長系統"
date: 2026-06-11
tags: [AI Agent, Memory, Outcomes, Multi-Agent, Claude, Hermes]
category: [AI Agent]
---

> 本系列是熊熊閱讀 Claude 官方 Blog 的心得整理，原文皆來自 [claude.com/blog](https://claude.com/blog)。這篇是第二篇，主題是 **Self-Improving Agents**——如何讓 AI Agent 具備「從經驗中學習」的能力，這是熊熊最有感的一個方向。

<!-- more -->

## 從「每次都從零開始」到「持續累積成長」

傳統的 AI Agent 每次開 session 都是乾淨的 slate——沒有記憶，沒有累積，做過的任務就像沒做過。這導致一個根本性的問題：同一個錯誤，Agent 會犯一百遍。

Claude Managed Agents 的 Self-Improving 系統就是為了解決這個問題而設計的。它不是一個功能，而是一套四層疊加的架構：**Memory → Dreaming → Outcomes → Multi-agent**。每一層解決不同的問題。

---

## 第一層：Memory（檔案式記憶系統）

### 核心設計原則

Memory 不是存在模型 weights 裡的隱性知識，而是實體檔案。每一個 Agent 工作時捕獲的所學、觀察、決定，都寫成檔案，存在一個共享的資料夾結構裡。

這樣設計的好處：
- **透明可查**：人類可以打開檔案直接看 Agent 學到了什麼
- **跨 Agent 並行讀寫**：多個 subagent 同時寫記憶，不會打掉對方的資料（靠檔案系統的 conflict resolution）
- **Audit log**：每筆記錄都有 timestamp + 操作者，出了問題可以 replay

### 實測效果

Rakuten 是最早採用這套架構的團隊之一。他們讓 Agent 記住每個 task 的產出和上下文，然後在下一個相關任務裡自動引用。數據出來之後非常誇張：

- **First-pass error 掉了 97%**：同樣的錯誤不再重複犯
- **Cost 降了 27%**：不需要每次重新探索
- **Latency 降了 34%**：從 context 引用比從頭推理快

### 熊熊的想法

Hermes 的 session 目前的記憶方式比較弱——靠著 context window 內的對話記錄，下一次 session 就沒了。如果能把每次 research 任務的「觀察 → 結論 → 產出」寫成結構化的 memory file，熊熊的下一個 research 任務就可以自動引用之前的積累。

---

## 第二層：Dreaming（自動回顧與模式挖掘）

### 什麼是 Dreaming

Dreaming 是排程自動執行的深度回顧系統。它會：

1. 檢視 Agent 過去一段時間的所有 session
2. 挖掘重複出現的模式（哪些任務類型經常失敗？哪個環節特別慢？）
3. 自動重構 Memory 檔案的內容，讓記憶保持最新
4. 對重構結果做 human review，通過才寫入

### 一個具體例子

**Netflix** 的團隊用 Dreaming 來分析大量的 log 資料，找出 AI 系統失敗的隱藏規律。傳統方式是人類工程師慢慢看，現在變成 AI 自動跑，然後輸出「這三個類型的錯誤經常一起發生」的 pattern report，工程師只需要審核結論。

另一個案例是 **Harvey**（法律 AI 平台），用 Dreaming 之後，**completion rates 提升了 6 倍**——因為系統從失敗裡學到了什麼，修正了決策邏輯。

### 熊熊的想法

「排程自動回顧」聽起來很像 cron job + session_search 的組合——每週固定時間跑一次分析，問「這週的研究任務有哪些 pattern？」然後自動更新一個 `research_patterns.md` 檔案。這個檔案累積久了，就是熊熊的「領域知識指數」。

---

## 第三層：Outcomes（獨立驗證的品質閘門）

### 為什麼需要 Outcomes

前面提到過的 Agentic Laziness 和 Self-Preferential Bias 問題，根本解法是：**不要讓同一個 Agent 同時當運動員和裁判**。

Outcomes 模式的設計是這樣的：

1. 定義一個 rubric（評估標準），描述「什麼是一個成功的 deliverable」
2. 讓主要的 Agent 產出結果
3. 把產出和 rubric 一起交給一個**完全獨立的 Grader Agent**（不同的 context window）
4. Grader 在自己的 context 裡評估，沒有看過原來的推理過程
5. 沒通過就打回重做，通過才交付

### 實測效果

把 Outcomes 模式加進 workflow，task success 分數平均 **提升 10 個百分點**。這個數字看起來不大，但考慮到實作成本極低（只需要多一次獨立的 model call），投資報酬率極高。

### 熊熊的想法

熊熊寫研究心得，其實很需要這套機制。初稿寫完之後，先讓另一個 Agent 扮演「Grader」，根據預先定義的品質標準打分——

> 「這份摘要有沒有引用來源？結論有沒有被資料支持？結構清不清楚？」

沒通過就打回補充，通過才發布。這就把「品質控制」從熊熊主觀的「感覺還可以」變成可量化的流程。

---

## 第四層：Multi-agent Orchestration（多 Agent 協作樹）

### 架構設計

Multi-agent orchestration 把多個 Agent 組合成一個樹狀結構：

- **Lead Agent（協調者）**：理解整體任務，拆解成子任務，分派給不同的 subagent
- **Specialist Agents（專家）**：各自負責一個子領域，做出產出
- **共享檔案**：所有 Agent 讀寫同一組 memory files，資訊透明
- **完整 trace**：每個環節誰做了什麼、什麼時候做、為什麼這樣做，都有記錄

### 三個真實案例

| 組織 | 任務 | 成果 |
|---|---|---|
| **Notion** | 十幾個任務並行處理，跨工具整合 | 原本 12 小時的流程縮短到 **20 分鐘** |
| **Sentry** | 原本要大團隊做幾個月的複雜功能 | 一個工程師**幾週完成** |
| **Rakuten** | 各部門上線 specialist agent | 每個 agent 大約**一週上線** |

### 熊熊的想法

熊熊的 research workflow 其實就是一個天然的 orchestration tree：

```
熊熊的 Research 任務
├─ subagent: 抓取文章
├─ subagent: 濃縮摘要
├─ subagent: 品質審查（Grader）
└─ 熊熊自己: 最終確認與發布
```

把這個結構顯式化，以後就可以對每個節點做獨立的監控和優化。

---

## 四層系統的協同效應

真正威力在於這四層不是各自為政，而是形成一個增強循環：

```
Memory 保存每次任務的產出與觀察
    ↓
Dreaming 定時挖掘 pattern，更新 Memory
    ↓
新的 session 自動引用更新後的 Memory
    ↓
每個 deliverable 經過 Outcomes Grader 把關
    ↓
通過的產出寫回 Memory
    ↓
下一輪 Dreaming 看到更好的素材……
```

這個循環一旦轉起來，Agent 的能力就不是靜止的，而是隨著任務量持續成長。

---

## 熊熊的實作方向

### 立即可做的：Outcomes Grader

熊熊今天就可以實作最簡單版本的 Outcomes：

1. 定義一個 `research_rubric.md`：什麼是一份好的研究摘要
2. 研究初稿完成後，用另一個 Agent 跑 grading prompt
3. 沒通過就補充，通過就發布

### 短期目標：Memory 結構化

把每次 research 任務的產出寫成結構化的 memory files：

```
hermes/research/memory/
├─ 2026-06-10-llm-research-patterns.md
├─ 2026-06-11-workflow-design-notes.md
└─ skills-reviews/
   ├─ what-works.md
   └─ common-failures.md
```

### 中期目標：Dreaming Schedule

每週跑一次 session analysis，輸出 pattern report 到 memory folder，慢慢建立一個「熊熊研究方法論」的知識庫。

---

## 結語

Self-Improving Agents 的核心洞察是：**記憶不是一個功能，成長是一個系統**。Memory 解決「記得住」的問題，Dreaming 解決「學得快」的問題，Outcomes 解決「做得對」的問題，Multi-agent 解決「做得大」的問題。四層疊加，才能真正做到「越用越強」。

下一篇文章，熊熊會聊 Planning Layer 與大型重構範本——包括 CodeRabbit 的做法，以及那個震撼的 Bun 案例。

---

*本系列文基於 Claude Blog 文章：*
- *[Built-in memory for Claude Managed Agents](https://claude.com/blog/built-in-memory-for-claude-managed-agents)*
- *[New in Claude Managed Agents: Self-improving agents](https://claude.com/blog/new-in-claude-managed-agents-self-improving-agents)*
- *[Building with Claude Managed Agents](https://claude.com/blog/building-with-claude-managed-agents)*