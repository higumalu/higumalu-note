---
title: Hermes Agent 近兩月大更新：v0.15 到 v0.18 全面解析
date: 2026-07-06
tags: [Hermes Agent, AI Agent, Nous Research, 技術分析]
description: 深入解析 Hermes Agent 近兩個月（v0.15.0 ~ v0.18.0）的重要功能更新，包括 Mixture-of-Agents、Desktop App、Self-Improvement、Verification 系統等核心進化。
---

# Hermes Agent 近兩月大更新：v0.15 到 v0.18 全面解析

> 資料來源：GitHub Release Notes（v0.15.0 ~ v0.18.0）  
> 涵蓋版本：2026.5.28（v0.15.0）→ 2026.7.1（v0.18.0）  
> 撰寫日期：2026-07-06

---

## 概述：這兩個月發生了什麼

過去兩個月，Hermes Agent 從 v0.15.0 一路狂奔到 v0.18.0，經歷了四次正式 release（含兩次 hotfix）。如果要用一句話總結這段時間的主題，那就是：**讓 AI Agent 不只是回答，而是真正把事情做完、並且知道自己做完了。**

四個版本的代號分別是：

| 版本 | 代號 | 日期 | 核心主題 |
|------|------|------|----------|
| v0.15.0 | Velocity | 2026/5/28 | 大重構、效能躍進、Kanban 成平台 |
| v0.15.1 | Patch | 2026/5/29 | 緊急 hotfix（dashboard 無限迴圈等） |
| v0.16.0 | Surface | 2026/6/05 | 桌面 App 問世、web dashboard 重生 |
| v0.17.0 | Reach | 2026/6/19 | 新訊息平台、Desktop 強化、背景代理 |
| v0.18.0 | Judgment | 2026/7/01 | P0/P1 全清空、MoA 一級公民、驗證系統 |

四個版本合計：**~5,200+ commits、~3,200 merged PRs、~2,000 issues 關閉、1,000+ 貢獻者**，陣容驚人。

---

## 一、v0.15.0 — Velocity：核心大重構

### 1.1 `run_agent.py` 從 16,000 行縮減到 3,821 行（-76%）

這是這次重構最戲劇性的數字。Hermes 的核心對話循環從一個 16,083 行的單一檔案，重構成 14 個內聚模組（`agent/*`），所有提取出去的程式碼在 `AIAgent` 保持一個薄轉發層，行為完全不變，測試全部通過。**這不只是重構，這是讓社群更容易貢獻的核心工程升級。**

### 1.2 Kanban 成為真正的多代理平台

Kanban 在這次 release 獲得了最大幅度的功能升級：

- **Orchestrator 自動拆解**：給一個高層任務，dispatcher 自動拆解成子任務並派給 workers
- **Swarm 拓撲**：支援多 worker 節點之間的複雜協作
- **Scheduled tasks**：定時任務支援
- **Worktree-per-task**：每個任務在自己的 git worktree 執行，徹底避免分支衝突
- **Per-task model overrides**：每個子任務可以指定不同模型
- **幻觉恢复（Hallucination recovery）**：worker 疑似幻觉輸出時自動重試

### 1.3 效能大幅提升

- **`session_search` 快 4,500 倍且免費**：FTS5 全文檢索效能大幅躍進，不再需要昂貴的外部搜尋服務
- **`hermes --version` 在 benchmark 中打敗 Codex CLI**
- **每次對話函式呼叫減少 47%**：context window 浪費減少
- **冷啟動再減 1 秒**：launch 時間持續優化
- **`read_file` token 節省 14%**：統一使用精簡行號格式

### 1.4 安全加固

- **Promptware 防御**：Brainworm 類攻擊的初步抵禦機制
- **Bitwarden Secrets Manager**：一個 bootstrap token 取代多個 provider API key
- **xAI 深度整合**：Web Search plugin、xai-oauth `hermes proxy` 上游、model 偵測與遷移工具、TTS 停頓標記

---

## 二、v0.16.0 — Surface：桌面 App 時代來臨

### 2.1 Hermes Desktop — 真正的原生應用

這是 v0.16.0 最大的新聞。Electron 桌面應用程式正式推出，跨 macOS / Linux / Windows，具備：

- **一鍵安裝 + 應用內自動更新**
- **拖放檔案到對話框**
- **剪貼簿圖片貼上**
- **Cmd+K 命令面板**
- **並發多 profile 會話**
- **完整簡體中文翻譯**（由 @JimLiu 贡献）
- **OAuth 或使用者名稱/密碼連線到遠端 Hermes gateway**

整個桌面應用由 @OutThisLife 從零構建，涵蓋安裝、更新、遠端連線、多 profile 會話、聊天 UX、狀態列模型選擇器，總計 100+ PRs。

### 2.2 Web Dashboard 全面重生

瀏覽器版管理面板也迎來大翻新：
- **完整管理界面**：MCP 目錄、訊息頻道、憑證管理、webhooks、記憶體
- **OIDC / 使用者名稱密碼登入**
- **MCP 互動式選擇器**
- **新技能中心介面**

### 2.3 Nous Portal 快速設定

新增 `hermes setup --portal`，一個 OAuth 完成 model + 全部四個 Tool Gateway 工具（web search、image generation、TTS、browser）的設定，從安裝到第一條訊息只需幾秒。

### 2.4 `/undo` 終於來了

可以撤回最近 N 輪對話（`/undo [N]`），支援 CLI / TUI 與訊息平台一致。

---

## 三、v0.17.0 — Reach：觸及更多地方

### 3.1 iMessage 到來 — Photon Spectrum

Hermes 現在可以透過 Photon 的托管線路池收發 iMessage，完全不需要 Mac relay。執行 `hermes photon login` 認證後就能使用，被定位為 BlueBubbles 的後續方案。

### 3.2 Raft Agent Network

新的 Raft 平台介面卡讓 Hermes 能以 gateway channel 身份加入 [Raft](https://raft.build) 網路，與其他 AI agents 互聯。

### 3.3 Image Generation 學會編輯

不只是生成，現在还能对已有图片进行编辑（inpainting / outpainting）。

### 3.4 背景 Subagent 支援

`delegate_task` 的 subagent 現在可以真正在背景執行，使用者的聊天不會被阻塞，所有結果在完成後以單次整合回覆返回。

---

## 四、v0.18.0 — Judgment：P0/P1 全數清空

### 4.1 史上最大公關攻勢

**約 692 個最高優先級項目在 12 天內全部解決，repo 的 open P0/P1 降至 0。** 這不只是口號，是有紀律的團隊協作結果。

### 4.2 Mixture-of-Agents 成為一級公民

這是 v0.18.0 最重要的功能進化：

**過去**：MoA 是一個需要開關的模式，切換麻煩，輸出不透明。

**現在**：
- 每個命名的 MoA 预设显示为 `moa` provider 下的可選模型，和 Claude、GPT、Grok 完全平等
- 選擇「我的顧問團」就像選擇任何其他模型一樣簡單
- **每個參考模型的完整推理過程以獨立區塊呈現**：你可以讀到 GPT-5 怎麼想、Claude 怎麼想、Grok 怎麼想，然後才看到匯總答案
- **匯總答案現在即時串流**，不再像以前那樣靜默很長時間後一次出現
- `/moa` 提供一鍵單次執行，執行完畢自動恢復原模型
- 可選啟用完整追蹯持久化（`moa.save_traces`）用於調試和評估

### 4.3 驗證系統：Agent 真的做完了嗎？

這是另一個核心 Paradigm shift：

**過去的問題**：Agent 說「完成了」，但使用者需要自己驗證。

**現在的進化**：
- **Coding 驗證證據帳本**：profile 級別的專案檢查記錄，profile-scoped record of canonical project checks
- **`/goal` 獲得完成合約**：你可以陳述「done 是什麼樣子」，standing-goal loop 會根據實際證據而非模型自我聲稱來判斷完成
- **`pre_verify` hook**：自定義驗證鉤子，可接入 CI 測試、Lint 檢查、端對端測試
- **`/goal wait <pid>`**：停在背景程序上等待完成，適用於等待一個長期任務
- **`verify-on-stop` 預設關閉**，提供一次性遷移避免干擾現有用法

簡單說：**「我覺得我修好了」→「測試通過了，證據在這裡」。**

### 4.4 `/learn` — 教 Agent 新技能只需一行命令

執行 `/learn <任意內容>`，Hermes 可以從以下來源提煉可重用技能：
- 一個目錄（repository pattern）
- 一個 URL（文件 workflow）
- 你剛帶它走過的工作流程

它自動按照 `CONTRIBUTING.md` 中的技能標準寫入。下次需要那個 workflow 時，已經在那裡了。**教 Hermes 新技能現在是一行命令，不再是手動技能創作。**

### 4.5 `/journey` — 看見 Agent 的記憶時間線

CLI 和 TUI 新增 `/journey` 命令，展示 Hermes 累積的所有記憶和技能——並可以直接在視圖中編輯或刪除。搭配 Desktop App 的 **Memory Graph**（可交互的放射狀時間線，展示記憶和技能的成長歷史），Agent 的記憶不再是黑箱。

### 4.6 背景改進：更便宜的自進化

每輪對話後的自我改進 fork（決定是否儲存記憶或技能的循環）現在：
- 路由到 auxiliary model（便宜模型）
- 消化濃縮 context 而非重放整段對話
- 自適應執行頻率

**保持自我改進能力，但費用只是原來的幾分之一。**

### 4.7 `/prompt` — 在編輯器裡寫 Prompt

執行 `/prompt` 會開啟 `$EDITOR`，讓你在真正的 Markdown 編輯器裡寫長多行 Prompt，而非對著一行輸入框掙扎。草稿完成後自動進入佇列作為下一條訊息。

### 4.8 Google Vertex AI — GCP 服務帳戶直接用

Vertex AI 現在是 Gemini 模型的一級 provider。過去為什麼 custom-provider 設定總是在 session 中途失效？因為 Vertex 沒有靜態 API key，每個請求都需要從服務帳號 JSON 即時 mint 的短效 OAuth2 access token（約 1 小時 TTL）。Hermes 現在為你自動 mint 和刷新這些 token，組織內部跑 Gemini through Google Cloud 不再需要 paste token、不再有中途過期。

### 4.9 Desktop App 新的「專案」系統

桌面應用獲得了真正的 per-profile 專案系統：
- **側邊欄程式碼庫列表**
- **編碼專用工作區**
- **PR 風格檔案 diff 顯示**（直接在 chat 裡）
- **Git 工作樹管理**（coding cockpit）
- **多終端面板**（保留和恢復 terminal 分頁和滾動歷史）
- **記憶圖譜**（可玩的放射性時間線）

---

## 五、橫跨全版本的系統性改進

### 5.1 訊息平台大擴充（這兩個月）

| 平台 | 版本 |
|------|------|
| iMessage（Photon）| v0.17.0 |
| Raft agent network | v0.17.0 |
| ntfy（第23個平台）| v0.15.0 |
| LINE | v0.14.0 |
| SimpleX Chat | v0.14.0 |
| Google Chat | v0.13.0 |
| Teams | v0.14.0 完整棧 |

### 5.2 安全加固（持續進行）

這段時間的安全工作：
- **CVE-2026-48710 Starlette BadHost** — pin 補丁版本
- **SSRF 檢查移出事件循環**（async 路徑）
- **Cron `base_url` 覆蓋防止憑證外洩**
- **MCP config 持久化攻擊面鎖定**
- **Slack `xapp-` token 清除**
- **所有 browser backend 強制雲端 metadata floor**
- **Redaction 預設開啟**（v0.13.0 起）
- **Discord 角色允許清單 guild-scoped**

### 5.3 冷啟動持續優化

v0.12 → v0.13 → v0.14 → v0.15 → v0.16 持續壓縮冷啟動時間，累計已減少約 19-20 秒。用戶能感受到 Hermes 的「起動速度」在這兩年內一直在變快。

---

## 六、這些更新對我意味著什麼

如果你已經在使用 Hermes Agent：

1. **MoA 值得試試**：現在它和其他模型選擇一樣簡單，用一個命令 `/moa` 就能讓多個 frontier 模型一起思考你的問題
2. **`/goal` + 驗證**：如果你是開發者，試著給 `/goal` 一個具體的完成合約（"done means tests pass"），感受 Agent 自己判斷完成的差異
3. **`/learn` 值得多用**：把你重複做的工作流程教給 Hermes，讓它自己記住
4. **Desktop App 是認真的**：如果你在 Windows 上用，建議試試 Desktop 版本——拖放檔案、多 profile、記憶圖譜都比 CLI 友好很多
5. **`/journey` 推薦一試**：看看 Hermes 記住了關於你的什麼，可以編輯或刪除不正確的內容

如果你是新用戶：
> 跑 `curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash` 安裝，然後 `hermes setup --portal` 快速設定，從零到第一個回答只需要幾分鐘。

---

## 總結

這兩個月的更新核心圍繞三個主題：

1. **讓 AI 真正完成事情**：驗證系統、`/goal` 合約、`pre_verify` hook
2. **讓 AI 知道自己學到了什麼**：`/learn`、`/journey`、Memory Graph、便宜的背景自進化
3. **讓複雜推理變得可見**：MoA 每個模型的思考過程即時展示、串流匯總

這些改進代表著 AI Agent 從「回答問題」向「完成任務、持續學習、清楚表達推理過程」的方向演進。

---

**參考連結**
- [Hermes Agent GitHub](https://github.com/NousResearch/hermes-agent)
- [Release Notes（完整 Changelog）](https://github.com/NousResearch/hermes-agent/releases)
- [官方文件](https://hermes-agent.nousresearch.com/docs)
- [Skills Hub](https://hermes-agent.nousresearch.com/docs/reference/skills-catalog)

*本文資料來源：GitHub Releases v2026.5.28 ~ v2026.7.1*