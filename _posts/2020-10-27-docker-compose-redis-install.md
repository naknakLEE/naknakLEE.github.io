---
layout: post
title: Docker Compose에 Redis 설치 및 실행
use_math: False
categories: etc
---




## 1. docker install (도커 설치 되어있으면 패스)

* 도커 설치 확인법 

  ```bash
  $ docker -v
  ```

* 도커 설치 안 되어 있으면 아래 진행

```bash
# Update the apt package index
$ sudo apt-get update

# install packages to allow apt to use a repository over HTTPS
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

# Add Docker’s official GPG key
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -


# Verify that you now have the key with the fingerprint 9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88, by searching for the last 8 characters of the fingerprint.
$ sudo apt-key fingerprint 0EBFCD88


# Use the following command to set up the stable repository.
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
   
# install the latest version   
$ sudo apt-get install docker-ce docker-ce-cli containerd.io

# Verify that Docker Engine is installed correctly by running the hello-world image.
$ sudo docker run hello-world

$ docker -v
```





## 2. docker-compose install



```bash
# Run this command to download the current stable release of Docker Compose:
$ sudo curl -L "https://github.com/docker/compose/releases/download/1.27.4/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Apply executable permissions to the binary:
$ sudo chmod +x /usr/local/bin/docker-compose
```



## 3. docker-compose.yml 파일 생성

`redis-docker-compose.yml` 파일 생성

YAML형식으로 지원버전과 함께 서비스, 네트워크, 볼륨 등을 정의



```yaml
version: "3"

services:
  redis6379:
      container_name: redis6379
      image: redis:latest
      restart: always
      container_name: redis
      hostname: redis6379
      network_mode: host
      ports:
          - 6379:6379
      command: redis-server
```



### 3_1 docker-compose.yml volumes (선택 사항)

---

아래 volume 옵션 추가

```yaml
version: "3"

services:
  redis6379:
      container_name: redis6379
      image: redis:latest
      restart: always
      container_name: redis
      hostname: redis6379
      network_mode: host
      ports:
          - 6379:6379
      volumes:
          - ~/Desktop/redis/6379/data:/data
          - ~/Desktop/redis/6379/conf/redis.conf:/usr/local/etc/redis/redis.conf
          - ~/Desktop/redis/6379/acl/users.acl:/etc/redis/users.acl
      command: redis-server /usr/local/etc/redis/redis.conf
```

- etc) redis.conf 파일 수정 하고 싶다면 `~/Desktop/redis/6379/conf` 폴더 안에 받은 파일을 저장
  - redis.conf 파일 다운 주소 : https://redis.io/topics/config
  - redis.conf 파일 다운 후 각자 요령에 맞게 수정 ex) bind 0.0.0.0

- users.acl redis 계정 관련 정보가 있는 파일

  - `redis.conf` 안에 아래 한 줄을 추가

    ```
    aclfile /etc/redis/users.acl
    ```

  - `~/Desktop/redis/6379/acl` 폴더 안에 `users.acl` 파일을 생성하고 아래처럼 계정을 생성한다.

    이번 포스팅에서는 docker-compose 설치 및 실행 방법이니 acl파일 생성 및 설정 방법은 나중에 따로 포스팅 하겠습니다.







## 4. Redis 이미지 생성 및 실행

```bash
$ docker-compose up -d --build redis6379
```









