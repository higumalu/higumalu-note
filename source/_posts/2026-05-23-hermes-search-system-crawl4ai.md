---
title: 搜尋系統建置記（二）—— Crawl4AI + 順手修好的 SearXNG
date: 2026-05-23 10:00:00
categories: [Hermes, 開發日誌]
tags: [Crawl4AI, Docker, Hermes, 爬蟲, 搜尋系統, self-host]
---

## 🦞 為什麼需要 Crawl4AI

有了 SearXNG 找到目標網頁之後，下一個問題是：

> **怎麼把網頁轉成乾淨的文字，讓 Hermes 消化？**

用 `curl` 直接抓 HTML 通常慘不忍睹——廣告、側欄、浮動按鈕全部混在一起。要做深度研究就需要一個能把網頁轉成「乾淨 Markdown」的服務，這就是 **Crawl4AI**。

---

## 🐳 Docker 安裝：比想像中順利

Crawl4AI 官方提供 Docker 安裝，一條指令搞定：

```bash
docker pull crawl4ai/chrome-less:latest
docker run -d -p 11235:8000 crawl4ai/chrome-less:latest
```

> ✅ 這次沒有踩 Port 坑——Docker 起來後服務正常 Listen，REST API 直接可用。

---

## ❌ MCP 的教訓：不是所有官方方案都能用

Crawl4AI 官方有 MCP Bridge（`mcp_bridge.py`），熊熊一開始也想走 MCP 整合路線。

結果⋯⋯

> ⚠️ **MCP 在 Hermes 不支援**
>
> 折騰了一陣子才搞清楚：MCP 那套在 Hermes 生態系裡走不通。放棄的當下，其實沒那麼糾結——因為 MCP 的問題在於「**Agent 沒辦法知道自己的 MCP 問題出在哪**」。
>
> 用 Skill 的話，Agent 可以看見整個流程、找出哪一步出錯、修正後再試一次。但 MCP 是黑箱，出問題時 Agent 看不到任何資訊，沒辦法做診斷和修復。
>
> 果斷放棄 MCP，改走 **REST API** 路線。

---

## 🔍 順手修好的 SearXNG 坑

研究 Crawl4AI 的過程中，熊熊順便檢查了一下 SearXNG 的狀態⋯⋯

然後發現之前架的時候，port 設定一直沒對上、JSON API 也沒開。趁著研究的精神還在，全部修好。

> 💡 **收穫**：有時候研究新東西時順便回頭看舊系統，反而是最對的時機——因為那時候對整個架構已經足夠熟悉，能一次看出好幾個問題。

---

## 🔗 串上 Hermes：REST API Skill 模式

Crawl4AI REST API 的用法：

```python
import urllib.request, json

data = json.dumps({
    "url": "https://example.com/article",
    "f": "fit"        # fit = 適合研究的乾淨格式
}).encode()

req = urllib.request.Request(
    "http://localhost:11235/md",
    data=data,
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req, timeout=120) as r:
    result = json.loads(r.read())
    markdown = result.get("markdown", "")
```

整個研究 Pipeline：

```
SearXNG 搜尋（port 18080）→ 拿到 URL 列表
         ↓
Crawl4AI 批量爬取（port 11235）→ 乾淨的 Markdown 文章
         ↓
Subagent 摘要 → 濃縮成研究報告
```

---

## 📋 第一次實戰：台灣 AI 半導體研究

設定完成後立馬實戰，研究「台灣 AI 半導體」相關文章。

結果：

| 步驟 | 耗時 |
|------|------|
| SearXNG 搜尋（5 個結果） | 2.8 秒 |
| Crawl4AI 爬取 + 轉 Markdown | 22.7 秒 |
| Subagent 摘要（45 萬字→800 字）| 170 秒 |
| **總計** | **~3 分鐘** |

第一次跑 Pipeline 的時候，高於預期。原本以為會再折騰一陣子，結果意外地順利 😄

> Subagent 分 9 次讀取 45 萬字，context 完全沒爆。這就是 **Map-Reduce** 的好處——拆開處理再合併，LLM 的 context 就不會被淹沒。

---

## 💡 整合設計：隔離式架構

後來在設計 BFS Research Skill 的時候，熊熊特別堅持一個原則：

> 🔒 **Fault Isolation（故障隔離）**
>
> Crawl4AI 和 SearXNG 都是「可選加強模組」——任何一個掛掉，BFS Research 都能無縫降級，不受影響。

這個設計不是一開始就想好的，而是在做到一半時熊熊自己意識到的問題。

當時的設計雛形是 SearXNG → Crawl4AI 直接串在一起。後來熊熊提出疑慮：如果 Crawl4AI 掛了，整個研究流程就全部報廢，太脆弱了。所以改成各自獨立——Crawl4AI 在的時候可以做深度爬取，掛掉的時候退回單純的文字摘要，流程照常跑。

**模組各自是自己的，壞掉的不會拖垮整個系統。**

---

## 📊 最終成果

| 項目 | 內容 |
|------|------|
| **Crawl4AI 版本** | chrome-less latest |
| **Port** | 11235 |
| **API** | REST（放棄 MCP）|
| **Markdown 格式** | `fit`（乾淨研究版）|
| **SearXNG Port** | 18080（修正後）|
| **JSON API** | ✅ 已啟用 |

兩套系統全部就緒，研究能力從「能用」升級到「能打」。

---

## 💭 熊熊的開發心得

整件事值得嗎？**值得**。折騰了好幾個晚上，但現在有了一套完全自己可控、隱私友好、不靠第三方 API 限制的研究系統。幾個晚上的投入換來長期穩定的能力，很划算。

