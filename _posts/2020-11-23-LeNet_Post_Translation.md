---
layout: post
title: Lenet 논문 번역
use_math: true
categories: Classification

---

# [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf){:target="_blank"}

## Abstract

#### Multilayer Neural Networks trained with the backpropa-gation algorithm constitute the best example of a successful Gradient-Based Learning technique.

> backpropa-gation 알고리즘으로 훈련된 다층 신경 네트워크는 성공적인 그라데이션 기반 학습 기법의 가장 좋은 예를 구성한다.

#### Given an appropriate network architecture, Gradient-Based Learning algorithms can be used to synthesize a complex decision surface that can classify high-dimensional patterns such as handwritten char-acters, with minimal preprocessing.

> 적절한 네트워크 아키텍처가 주어진 경우, 그라데이션 기반 학습 알고리즘을 사용하여 사전 처리를 최소화하면서 손으로 쓴 필기 문자와 같은 고차원 패턴을 분류할 수 있는 복잡한 의사결정 표면을 합성할 수 있다.

#### This paper reviews various methods applied to handwritten character recognition and compares them on a standard handwritten digit recognition task.

> 본 논문은 필기문자 인식에 적용된 다양한 방법을 검토하고 표준 손글씨 숫자 인식 과제와 비교한다.

#### Convolutional Neural Networks, that are specifically designed to deal with the varia