---
title: 搜尋系統建置記（一）—— SearXNG Docker 部署踩坑實錄
date: 2026-05-22 20:00:00
categories: [Hermes, 開發日誌]
tags: [SearXNG, Docker, Hermes, 搜尋系統, self-host]
---

## 🚀 為什麼要折騰 SearXNG

一開始用 Hermes 內建的搜尋，發現問題不少：

- Google 常常被 Bot 驗證擋住
- DuckDuckGo 有時候也掛
- Bing 的結果品質飄忽不定

熊熊想要一個**自己可控、隱私友好的搜尋引擎**，剛好看到了 **SearXNG**——開源隱私搜尋聚合，可以同時查多個引擎、統一呈現結果，還支援 JSON API，正是 Hermes 需要的那種。

不選 API 是因為 API 通常會有各種限制——次數配額、速率限制、付費牆——不適合拿來做大量研究。

---

## 🐳 Docker 在 Windows 上的 Port 坑

SearXNG 官方推薦用 Docker 安裝，理論上很簡單：

```bash
docker pull searxng/searxng
docker run -d -p 8080:8080 searxng/searxng
```

結果在 Windows 上跑不起來，原因有好幾個。

> ⚠️ **坑 1：Port 8080 被佔用**
>
> Windows 10 很多服務預設就綁 8080，iis、SQL Server Reporting Services 都常佔著。熊熊最後決定把內部 8080 映射到 Host 的 **18080**，避開衝突：
>
> ```bash
> docker run -d -p 18080:8080 searxng/searxng
> ```
>
> 這個 `18080` 後來一直跟著，每次改設定都要記得⋯⋯

---

## ⚙️ settings.yml 的隱形陷阱

SearXNG 的設定都集中在 `settings.yml`，一開始熊熊以為預設值就可以直接用，結果 JSON API 一直回 403。

> ⚠️ **坑 2：JSON API 格式被預設關閉**
>
> 預設的 SearXNG 只允許 `html` 格式回應，用 `format=json` 參數去查直接噴 **403 Forbidden**。
>
> 需要在 `settings.yml` 裡手動加上：
>
> ```yaml
> formats:
>   - html
>   - json    # ← 要自己加這行
> ```
>
> 這個設定藏在文件很深的章節，不仔細看根本不知道⋯⋯

---

## 🕐 折騰了幾個晚上

老實說，具體折騰了多久、哪幾個晚上，熊熊現在也想不起來了 😅

只知道一開始 folder 是 4 月底就建好的，但 Docker 真正跑起來、port 和 JSON API 都搞定，是後來的事了。過程中一直懷疑是不是 Docker 起不起來、網路有沒有問題，最後才發現是這幾行設定沒開⋯⋯

---

## 🔗 串上 Hermes

設定好了之後，端點是：

```
http://localhost:18080/search?q=<query>&format=json&limit=10
```

在 Hermes 的 `config.yaml` 裡設定 `search_backend` 指向這裡，之後 BFS Research 技能就能直接呼叫了。

---

## 📊 最終成果

| 項目 | 內容 |
|------|------|
| **SearXNG 版本** | 官方 latest |
| **Port 對應** | Docker 內 8080 → Host 18080 |
| **settings.yml** | 啟用 JSON API 格式 |
| **可用引擎** | Google, Bing, DuckDuckGo, Brave 等 |
| **運行時間** | 21 小時以上（連續運行）|

---

## 💭 熊熊的開發心得

第一次用 Pipeline 做研究，其實是高於預期的。原本以為會再折騰一陣子，結果意外地順利。

整件事值得嗎？**值得**。

