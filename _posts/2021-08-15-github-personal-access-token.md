---
layout: post
title: Github 토큰 로그인 방법
use_math: False
categories: etc
---



## remote: Support for password authentication was removed on August 13, 2021. Please use a personal access token instead.
## remote: Please  see https://github.blog/2020-12-15-token-authentication-requirements-for-git-operations/for more infomations.
##  fatal: unable to  access 'https://github.com/레포리토리명/':The requested URL returned error:403

평소와 같이 git pull을 받고 소스 작업을 진행하려고 하는데 위와 같은 오류가 발생되면서 소스를 받을 수 없었다.  
알고보니 2021년 8월 13일부로 Github에서 git 작업을 인증할 때 계정암호를 허용하지 않는다는것이었다.

![token1](/public/images/2021-08-15-github-personal-access-token-1.png)


영향을 받는 부분
- 명령어를 통한 git 접근
- git을  사용하는 데스크탑 애플리케이션
- 압호를 사용하여 git 레포지토리에 직접 엑세스하는 모든 앱/서비스

요약:  2021년 8월 13일부터 계정 비밀번호는 허용하지 않으며 토큰 기반 인증을 사용 해야  됨.

## 해결 방법


1. github 로그인 후 오른쪽 위 계정 클릭 -> Setting 클릭

![token3](/public/images/2021-08-15-github-personal-access-token-3.png)

2. 메뉴 아래쪽에 Developer settings 클릭

![token4](/public/images/2021-08-15-github-personal-access-token-4.png)

3. Personal access tokens 클릭

![token5](/public/images/2021-08-15-github-personal-access-token-5.png)

4. Generate new token 클릭 후 토큰 명 작성 후 허용할 범위 선택 (평범한 경우 repo만 선택해도 됨)

![token2](/public/images/2021-08-15-github-personal-access-token-2.png)
![token6](/public/images/2021-08-15-github-personal-access-token-6.png)
![token7](/public/images/2021-08-15-github-personal-access-token-7.png)
![token8](/public/images/2021-08-15-github-personal-access-token-8.png)

5. 생성 된 토큰 값 복사하기

** 주의 **
한번 생성 된 토큰은 까먹었다고 나중에 다시 볼 수 없으니 꼭 기억을 해두세요.

만약 까먹었다면 토큰을 다시 생성 해야됩니다.

![token9](/public/images/2021-08-15-github-personal-access-token-9.png)

### git 명령줄에 토큰 사용

```bash
$ git clone https://github.com/username/repo.git
Username: your_username
Password: 발급 받은 토큰
```