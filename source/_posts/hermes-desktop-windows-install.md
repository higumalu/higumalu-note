---
title: Windows 安裝 Hermes Agent 桌面版：官方懶人指南（2026 最新）
date: 2026-06-12 10:00:00
tags: [Hermes Agent, AI, 教學, Windows, 桌面版]
categories: [AI 工具]
---

# Windows 安裝 Hermes Agent 桌面版：官方懶人指南（2026 最新）

> 更新時間：2026-06-12｜適用版本：Hermes Agent v0.16.0｜來源：[Nous Research 官方](https://hermes-agent.nousresearch.com/desktop)

---

## TL;DR — 60 秒懶人包

**Windows 10/11 用戶現在可以直接下載 `.exe` 安裝了！**

1. 去 [hermes-agent.nousresearch.com/desktop](https://hermes-agent.nousresearch.com/desktop) 下載 Windows 安裝包
2. 一路點「下一步」完成安裝
3. 啟動軟體，填入你的 LLM API Key
4. 開始跟你的 AI 助理聊天 🎉

懶得動滑鼠？用 PowerShell 一行搞定（實驗性）：

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

---

## Hermes Desktop 是什麼？

Hermes Desktop 是 [Nous Research](https://nousresearch.com) 官方推出的桌面客戶端，讓你不需要碰指令列，就能用上 Hermes Agent 的全部能力。

**它不是另一套聊天機器人** — 它就是你之前在終端機用 CLI 跑的 Hermes Agent，只是包了一層漂亮的 GUI 外殼。兩者共用同一套設定、記憶、技能、對話紀錄，隨時可以切換。

### 核心功能亮點

- **即時串流對話**：工具呼叫（查網頁、搜尋資料、操作檔案）全部即時顯示進度
- **MCP 支援**：一鍵連接多個 MCP 伺服器，擴充你的 AI 能力
- **跨平台訊息整合**：Telegram、Discord、Slack、WhatsApp、Signal 等 15+ 訊息平台同一個介面管理
- **自我進化學習迴圈**：複雜任務完成後自動建立技能（Skill），下次遇過類似問題直接用
- **長期記憶**：FTS5 搜尋 + LLM 摘要，跨 session 的脈絡自動記住
- **任何模型都能接**：OpenAI、Claude、DeepSeek、Ollama 本地模型、OpenRouter 等

> **MIT 授權，完全開源**，資料預設存在本機，隱私自己掌控。

---

## 三種 Windows 安裝方式

### 選項一：直接下載安裝（✅ 推薦給大多數人）

這是 2026 年 Hermes v0.16.0 新增的官方懶人功能，專為 Windows 用戶設計。

**步驟：**

1. **下載安裝檔**
   前往官方下載頁：
   [https://hermes-agent.nousresearch.com/desktop](https://hermes-agent.nousresearch.com/desktop)
   
   點擊 Windows 區塊的「Download」按鈕，下載 `Hermes-Setup.exe`（約 113 MB）。

2. **執行安裝程式**
   雙擊 `.exe` 檔，Windows SmartScreen 可能跳出警告，點「仍然執行」即可。
   安裝路徑預設：`%LOCALAPPDATA%\hermes`

3. **首次啟動設定**
   啟動 Hermes Desktop 後，會引導你：
   - 選擇要**本地模式**（軟體自己裝 Hermes runtime）或**遠端模式**（連到已架好的 server）
   - 設定 LLM API Key（支援 Nous Portal、OpenRouter、OpenAI、Anthropic 等）
   - 設定訊息頻道（可略過，稍後再設定）

4. **完成，開始使用！**
   安裝程式已幫你處理好所有依賴，包括 Python、Node.js、ffmpeg 等，不需要自己另外安裝。

---

### 選項二：從原始碼編譯（進階玩家）

如果你想用最新的尚未發布功能，或者想要控制編譯過程，可以從 source 編譯。

**事前準備：**

- Volta（Node.js 版本管理器）：[volta.sh](https://volta.sh)
- Git
- 約 5–10 GB 磁碟空間

**編譯步驟：**

```bash
# 1. 安裝 Volta（尚未安裝的話）
winget install Volta

# 2. 導向 Desktop 原始碼目錄
cd "C:\Users\<你的使用者名稱>\AppData\Local\hermes\hermes-agent\apps\desktop"

# 3. 在 package.json 加入 Volta 節點版本綁定
#    （編輯 package.json，加入以下內容到根層級）
{
  "volta": {
    "node": "22.22.3",
    "npm": "10.9.0"
  }
}

# 4. 安裝 Windows 原生 binding（編譯所需）
npm install @rolldown/binding-win32-x64-msvc
npm install @tailwindcss/oxide-win32-x64-msvc

# 5. 安裝專案依賴
npm install

# 6. 建置 Electron 介面
npm run build

# 7. 打包成 Windows 安裝檔
npm run dist:win
```

編譯完成後，輸出檔案在 `apps/desktop/release/` 目錄：
- `Hermes-X.X.X-win-x64.exe` — NSIS 安裝程式（推薦）
- `Hermes-X.X.X-win-x64.msi` — MSI 安裝程式
- `win-unpacked/Hermes.exe` — 免安裝可攜版

---

### 選項三：WSL2 + CLI 模式（傳統穩定路線）

官方對原生 Windows 的支援目前處於「公開預覽」階段（public preview），如果選項一下載後遇到問題，WSL2 是更穩定的替代方案。

**適用情境：** 追求最大穩定性、避免踩坑的用戶

#### 步驟 1：啟用 WSL2

以**系統管理員身份**開啟 PowerShell，執行：

```powershell
wsl --install
```

安裝完成後**重開機**，WSL 會引導你設定 Ubuntu 使用者名稱和密碼。

#### 步驟 2：安裝 Hermes Agent

在 Ubuntu（WLS2）終端機內執行官方安裝腳本：

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

腳本會自動幫你安裝所有東西（uv、Python 3.11、Node.js、ripgrep、ffmpeg 等），**完全不需要 sudo**，也不需要預先安裝任何軟體。

#### 步驟 3：啟動並設定

```bash
source ~/.bashrc      # 載入環境
hermes                # 啟動互動式 CLI
```

#### 步驟 4：順帶安裝桌面 GUI（非必須）

CLI 安裝完成後，若想要桌面版 GUI，可在 WSL2 內執行：

```bash
hermes desktop
```

---

## 安裝後的初始設定

### 設定 LLM API Key

Hermes Desktop 安裝完後，需要告訴它要用哪個 AI 模型。支援的 provider：

| Provider | 優點 | 適合族群 |
|----------|------|----------|
| [Nous Portal](https://portal.nousresearch.com) | 300+ 模型，整合最好 | 想要完整功能的人 |
| [Anthropic Claude](https://console.anthropic.com) | 工具呼叫最準確 | 研究、分析用途 |
| [OpenAI](https://platform.openai.com) | 生態成熟 | 習慣 OpenAI 生態的人 |
| [Ollama](https://ollama.com)（本地） | 完全免費、離線可用 | 注重隱私的人 |
| [DeepSeek](https://platform.deepseek.com) | 成本低 | 成本敏感型用戶 |

設定 API Key 的方式（CLI 模式）：

```bash
hermes config set ANTHROPIC_API_KEY sk-ant-xxxxx
hermes config set model anthropic/claude-sonnet-4
```

在桌面版 GUI 內則是去 **Settings → Providers** 填入即可。

### 驗證安裝是否成功

在 CLI 模式輸入：

```bash
hermes doctor
```

這會檢查所有依賴是否正確安裝、API Key 是否有效，出現 ✅ 就代表一切正常。

---

## YouTube 影片資源

以下是社群中品質較好的 Hermes Agent 安裝教學影片，涵蓋不同切入點：

| 標題 | 語言 | 內容 | 連結 |
|------|------|------|------|
| Hermes Agent 完整安裝指南 | 英文 | 涵蓋 Windows WSL2、macOS、API 設定 | [Grand Linux](https://www.youtube.com/@grandlinuxchannel6399) |
| Hermes Desktop 新功能展示 | 英文 | v0.15.2 Desktop 串流對話、MCP 設定 | [MarkTechPost 報導](https://www.marktechpost.com/2026/06/03/nous-research-releases-hermes-desktop-a-native-cross-platform-front-end-for-hermes-agent-v0-15-2-with-streaming-tool-output) |
| Windows WSL2 完整架設教學 | 英文 | OpenClaw Launch 整理的疑難排解樹 | [openclawlaunch.com](https://openclawlaunch.com/guides/hermes-agent-windows) |

> 🔍 想找更多？去 YouTube 搜尋 `Hermes Agent install` 或 `Hermes Agent Windows WSL2`，新影片每週都有。

---

## 常見問題

### Q1：安裝時跳出「Windows 已保護你的電腦」怎麼辦？

這是 Windows SmartScreen，屬於正常現象。點「仍然執行」即可，Hermes 是 MIT 開源專案，原始碼公開可審計。

### Q2：Native Windows 安裝失敗怎麼辦？

如果選項一（直接下載）有問題，強烈建議改走 **WSL2 路線**（選項三）。根據官方說法，WSL2 是經過最充分測試的路徑。

### Q3：WSL2 安裝卡在「正在安裝」很久，正常嗎？

正常。WSL2 首次安裝需要下載 Ubuntu 映像，視網速約需 5–15 分鐘，請耐心等待，不要關閉終端機。

### Q4：可以只裝 CLI，不裝桌面版嗎？

可以。WSL2 路線預設就是純 CLI，桌面 GUI 是可選的附加功能。

### Q5：Hermes Desktop 和 CLI 可以同時用嗎？

可以。兩者共用同一份設定（`~/.hermes/config.yaml`）和對話紀錄，切換時 session 無縫接軌。

### Q6：安裝後 Agent 沒反應怎麼辦？

檢查三件事：
1. API Key 有沒有正確設定（`.env` 檔或 Settings 頁面）
2. 網路能不能連到你的 LLM provider
3. 執行 `hermes doctor` 看診斷輸出

---

## 官方資源整理

| 資源 | 連結 |
|------|------|
| 官方網站 | [hermes-agent.nousresearch.com](https://hermes-agent.nousresearch.com) |
| 官方桌面下載 | [hermes-agent.nousresearch.com/desktop](https://hermes-agent.nousresearch.com/desktop) |
| 官方文件 | [hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs) |
| GitHub 原始碼 | [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) |
| 社群桌面版（非官方） | [github.com/fathah/hermes-desktop](https://github.com/fathah/hermes-desktop) |
| Discord 社群 | [discord.gg/nousresearch](https://discord.gg/nousresearch) |
| Skills Hub | [agentskills.io](https://agentskills.io) |

---

## 總結

**Windows 用戶的最好時機到了。** 2026 年 Hermes v0.16.0 正式推出 `.exe` 安裝程式，不需要 WSL2、不需要指令列技術，60 秒下載、3 分鐘設定就能開始用。

如果你是：

- **一般用戶** → 直接下載 `.exe`，最省事
- **進階用戶 / 開發者** → 從原始碼編譯，完全掌控
- **追求穩定性** → WSL2 + CLI，最經過考驗的路徑

有任何問題歡迎在 Discord 社群提問，或到 GitHub 開 Issue。祝安裝順利 🎉

---

*本文資料來源：官方文件、[Hermes Atlas 安裝指南](https://hermesatlas.com/guide/install)、[ClawSimple 安裝文](https://clawsimple.com/en/blog/hermes-agent-install-config-use-guide)、[MarkTechPost v0.16.0 報導](https://www.marktechpost.com/2026/06/03/nous-research-releases-hermes-desktop)。*