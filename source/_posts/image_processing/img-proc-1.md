---
title: histogram practice with python
date: 2021-10-22 08:48:13
tags: 
- image processing
- histogram
- image
---
Author：滷棕熊

## 輸入影像並轉為灰階
(單通道)

```py
import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread('sono01.bmp')  # input img
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #cover2onechannel
```
## 用calcHist函數計算直方圖
```
cv2.calcHist(影像, 通道, 遮罩, 區間數量, 數值範圍)
```
參數說明
- 影像：2D,N通道的影像 要用中括弧包起來
- 通道：若為RGB影像需填入channel的代表數,本例為單一通道影像
- 遮罩：需為一個大小與影像相同的8-bit矩陣,本例無使用遮罩
- 區間數量：總共有幾個線條
- 數值範圍：需要計算的pixel value範圍

```py
hist = cv2.calcHist([gray], None, None, [256], [0, 255])
hist = np.ndarray.flatten(hist)
```


## 輸出影像
在imshow中須設定color map參數
```
plt.cm.gray
```

```py
plt.figure(dpi=150)
plt.imshow(gray, plt.cm.gray)
plt.show()

plt.figure(dpi=150)
plt.bar(range(1,257), hist)
plt.show()
```