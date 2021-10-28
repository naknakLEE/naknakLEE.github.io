---
layout: post
title: 코드 컴플리트2 - 2부 고품질 코드작성
use_math: False
categories: codecomplete
---

안녕하세요 낙낙이입니다.

이 포스팅은 코드 컴플리트2 - 더 나은 소프트웨어 구현을 위한 실무 지침서의 요약본입니다.

제가 보면서 중요하다고 생각되는 부분, 메모로 남겼으면하는 하는 부분을 적어놨습니다.

보면서 궁금하신 있으시면 언제든 댓글이나 연락주세요!


![goofys1](/public/images/2021-08-13-code-complete-1.jpg)


## 좋은 함수 이름

- 함수이 하는 모든것을 표현하라
- 의미가 없거나 모호하거나 뚜렷한 특징이 없는 동사를 사용하지 말라.
  - HandleCalculation()이나 ProcessInput(), DealWithOutput()의 함수명은 무슨 일을 하는지 말해주지 않는다.
- 함수 이름을 숫자만으로 구분하지 말라.
  - part1(), part2() 사용 x
- 함수 이름의 길이에 신경 쓰지 마라.
  - 저걱절한 길이는 9 ~ 15줄이다. 하지만 전반적으로 함수 이름은 "명료함"에 초점을 맞춰야 하고, 따라서 이름의 길이에 제약을 받지 앟고 이해하기 쉽게 이름을 지어야 한다.
- 함수의 이름을 지을 때는 리턴 값에 관해서 설명하라.
  - cos(), customerId.Next(), printer.IsReady(), pen.CurrentColor() 등
- 프로시저의 이름을 지을 때 확실한 의미가 있는 동사를 객체 이름과 함께 사용하라.
  - PrintDocument(), CalcMonthlyRevenues(), CheckOrderInfo()
- 반의어를 정확하게 사용해라.
  - add/remove
  - increment/decrement
  - open/close
  - begin/end
  - insert/delete
  - show/hide
  - create/destroy
  - lock/unlock
  - source/target
  - first/last
  - min/max
  - start/stop
  - get/put
  - next/previous
  - up/down
  - get/set
  - old/new
- 공통적인 연산을 위한 규약을 만들어라.
- 