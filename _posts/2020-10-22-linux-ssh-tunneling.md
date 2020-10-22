---
layout: post
title: ubuntu(linux) ssh 터널링, ssh port forwarding
use_math: true
categories: etc
---



안녕하세요 낙낙이입니다.



오늘은 ssh 터널링, sshport forwarding에 대해서 알아보겠습니다.

고객사에 설치한 코드가 잘 동작하는지, 혹시나 오류는 없는지, 오류가 났으면 실시간 대응을 해야 되는데 고객사에서 방화벽으로 막아놔서 접속을 할 수 없더라고요.

보통은 ssh접속을 통해서 원격으로 접속해서 이것 저것 작업들을 했었는데 이번에는 방화벽때문에 막혀서 접근 조차 할 수 없더라고요.

그래서 찾아보니 ssh 터널링, vpn, 각종 원격 프로그램등을 알아보다가 ssh 터널링이 제가 가장 편하게 사용 할 수 있는거 같아서 ssh 터널링을 선택했어요





## SSH 터널링, SSH port Forwarding 이란?



- ssh 클라이언트와 서버 사이에 연결이 이루어지면 이를 터널링이라고 합니다.
- 여기에 포트 포워딩(Port Forwarding)이랑 기술을 더해서 다른 어플리케이션에 접근을 할 수 있습니다.
- ssh 특성 상 ssh 터널링을 통해 전달되는 데이터는 모두 암호화 됩니다.
- 터널링을 통해서 방화벽을 우회 할 수 있습니다.
  - ssh가 방화벽에 의해 차단 당하면 안됩니다. 





## 터널링의 종류



SSH 포트 포워딩은 연결을 수립하는 주체가 누구냐에 따라 Local과 Remote로 구분 할 수 있다. 

- A pc에 접속하고 싶은데 방화벽으로 막혀 있다 -> Remote port forwarding

- A pc에 접속은 가능한데 A pc의 로컬 서비스(ex. 127.0.0.1:80)를 이용하고 싶다. -> Local port forwarding

  





### Local port forwarding



사용법

```bash
$ ssh -L 포트번호1:호스트명:포트번호2 서버명
```



예시 )

A pc:

- 외부에서 ssh 접속이 가능함.

B pc:

- A pc로 ssh 접속이 가능함
- A pc의 있는 로컬호스트를 접속 하고 싶음 ex) A pc의 127.0.01:80



1. A pc에서의 작업

   ```bash
   $ ssh -L 9999:localhost:80 test@192.168.0.X
   ```

2. B pc에서의 작업

   ```bash
   $ ssh test@localhost -p 9999
   ```

   





### Remote port forwarding



사용법

```bash
$ ssh -R 포트번호1:호스트명:포트번호2 서버명
```



제가 자주 쓰는 기능입니다.



예시 )

A pc : 

- 방화벽으로 외부에서는 접속 할 수 없음
- **ssh 아웃 바운드는 열려있음. 즉, B pc로 ssh 접속이 가능함**

B pc : 

- 현재 내 PC
- A pc에 접속해서 작업을 하고 싶지만 방화벽 때문에 막혀있음



1. A pc에서의 작업

```bash
$ ssh -R 9999:localhost:22 test@192.168.0.X
```

#### Info.

​	9999 : B pc에서 접속 할 때의 포트 

​	localhost:22 : 9999번 포트로 접속하면 localhost:22번 포트로 연결 해 주겠다(= ssh)

​	test@192.168.0.x 이거는 B  pc의 정보





2. B pc에서의 작업

```bash
$ ssh test@localhost -p 9999
```

#### info

​	test : A pc의 username

​	localhost : B pc의 로컬 호스트를 가져가면 된다(대부분을 localhost로만 쓰면 될듯)

​	9999 : A pc에서 포트 9999번 포트를 열어놨기 때문에 9999로 접속





참고 자료 

[1] https://www.hanbit.co.kr/network/category/category_view.html?cms_code=CMS5064906327

[2] https://www.ssh.com/ssh/tunneling/example

[3] https://ithub.tistory.com/328