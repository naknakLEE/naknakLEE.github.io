---
layout: post
title: 우분투 18.04 (리눅스) 부팅 시 자동 파이썬 실행
use_math: true
categories: etc
---

## 서비스를 실행 할 스크립트 작성
   - 위치 : 상관없음
	 - ex) /home/test/test.py

```python
# location : /home/test/test.py
# HOWTOUSED : python3 test.py

import os
import time

while:
    time.sleep(1)
	print("Hello world")

```

## systemctl service 등록


### systemd에 Service 등록


1. /etc/systemd/system/서비스이름.service 파일을 만든다.
	```bash
	# 터미널 열고

	$ sudo vi /etc/systemd/system/서비스이름.service
	```
	```bash
	# 예시

	$ sudo vi /etc/systemd/system/test_service.service
	```



2. /etc/systemd/system/서비스이름.service를 만들고 해당 파일에 아래 작성 후 저장

	```bash
	[Unit]
	Description= 서비스 설명
	After= 해당 유닛이 시작된 이후 나열된 유닛이 실행

	[Service]
	Type=idle
	WorkingDirectory= 해당 명령어가 working할 경로 # 별도의 지정이 없으면 유닛은 "/" 디렉토리를 작업 디렉토리로 사용한다. 특정 디렉토리에서 실행해야하는 프로세스에서 필요하다. 
	ExecStart=실행할 명령어 # 절대경로
	StandardOutput= 로그 저장 파일 경로 # 절대 경로
	StandardError= 에러 로그 저장 파일 경로 # 절대 경로


	[Install]
	WantedBy= "systemctl enable" 명령어로 유닛을 등록할때 등록에 필요한 유닛을 지정
	```
	```bash
	# 예시
	[Unit]
	Description=My Script Service
	After=multi-user.target

	[Service]
	Type=idle
	WorkingDirectory=/home/test
	ExecStart=/usr/bin/python3 -u /home/test/test.py // 절대경로
	StandardOutput=file:/home/test/myscript.log // 절대 경로
	StandardError=file:/home/test/myscript_error.log // 절대 경로


	[Install]
	WantedBy=multi-user.target
	```

3. 권한 수정
   ```bash
   # 터미널 열고

   $ sudo chmod 755 서비스 경로
   ```

   ```bash
   # 예시
   
   $ sudo chmod 755 /etc/systemd/system/test_service.service
   ```


4. 서비스 등록 및 시작

	```bash
	# 터미널 열고

	$ sudo systemctl daemon-reload
	$ sudo systemctl enable 서비스이름
	```

	```bash
	# 예시 

	$ sudo systemctl daemon-reload
	$ sudo systemctl enable 서비스이름
	```
5. 재부팅 및 확인
   ```bash
   # 재부팅 후
   $ sudo systemctl status 서비스이름 

   ```

## etc) systemctl에 로그를 확인하는 방법

### journalctl을 이용해 systemd 로그 확인하기

1. 특정 유닛(서비스)의 로그를 확인
	```bash
	# u 옵션을 사용
	journalctl -u [systemd unit name]
	```

2. 로그를 실시간으로 확인하며 트래킹
   ```bash
   # -f 옵션을 사용
   journalctl -f [systemd unit name]

   ```
