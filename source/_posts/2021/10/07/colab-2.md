---
title: google colab (1) connect to drive 
date: 2021-10-07 23:26:49
tags:
- colab
- drive
categories:
- colab
---

### 連結到Google Drive
&nbsp;
```py
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse

from google.colab import auth
auth.authenticate_user()

from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()

import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}

```
```
E: Package 'python-software-properties' has no installation candidate
Selecting previously unselected package google-drive-ocamlfuse.
(Reading database ... 155047 files and directories currently installed.)
Preparing to unpack .../google-drive-ocamlfuse_0.7.26-0ubuntu1~ubuntu18.04.1_amd64.deb ...
Unpacking google-drive-ocamlfuse (0.7.26-0ubuntu1~ubuntu18.04.1) ...
Setting up google-drive-ocamlfuse (0.7.26-0ubuntu1~ubuntu18.04.1) ...
Processing triggers for man-db (2.8.3-2ubuntu0.1) ...
Please, open the following URL in a web browser: https://accounts.google.com/o/...
··········
Please, open the following URL in a web browser: https://accounts.google.com/o/...
Please enter the verification code: Access token retrieved correctly.

```
&nbsp;
執行之後點連結會產生一段驗證碼
貼到下方的格子中案Enter進行驗證
*一共需要驗證兩次驗證瑪*
&nbsp;
&nbsp;
```py
!mkdir -p Drive
!google-drive-ocamlfuse Drive

```
```
['.config', 'Drive', 'adc.json', 'sample_data']
```
在虛擬機開一個資料夾
名稱為'Drive'視為個人雲端硬碟的根目錄
&nbsp;
```py
os.chdir('Drive/colab_test')
print(os.listdir())
```
```
['GPUtest.ipynb', 'Untitled0.ipynb']
```
&nbsp;
切換到目前的工作資料夾
列出所有檔案

![Imgur](https://imgur.com/9tQ7FyJ.jpg)

&nbsp;
&nbsp;

