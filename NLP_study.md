NLP / tensorflow2
================



> # Introduction
>
> `텐서플로 2와 머신러닝으로 시작하는 자연어처리` 교재를 공부한 내용을
> 기록한다.

> ### Package loading

``` r
library(dplyr)
library(reticulate)
```

``` python
import tensorflow as tf
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt
```

> ## **Tensor?**

텐서란 N차원 매트릭스를 의미. 말 그대로 텐서를 Flow한다는 것은 데이터
흐름 그래프(Data flow graph)를 사용해 수치 연산을 하는 과정을 의미.
그래프의 노드는 수치연산, 변수, 상수를 나타내고, 에지는 노드 사이를
이동하는 다차원 데이터 배열(tensor)를 나타낸다.

> ## **tf.keras.layers**

텐서플로를 이용해 하나의 딥러닝 모델을 만드는 것은 마치 블록을 하나씩
쌓아서 전체 구조를 만들어 가는 것과 비슷하다. 따라서 쉽게 블록을 바꾸고,
여러 블록들의 조합을 쉽게 만들 수 있다는 것은 텐서플로의 큰 장점이다.
텐서플로에서는 블록 역할을 할 수 있는 모듈의 종류가 다양한데, 우리는
텐서플로 2.0 이후 모듈을 통합해 표준으로 사용하고
있는`tf.keras.layers`모듈에 대해 알아본다.

> ## **tf.keras.layers.Dense**

Dense란 신경만 구조의 가장 기본적인 형태를 의미한다.

$$
\\underline{y=f(Wx+b)}
$$

위 식에서 x와 b는 각각 입력벡터, 편향 벡터이며 W는 가중치 행렬이다.즉
가중치 함수에서 입력 벡터를 곱한 후 편향을 더해준다. 그리고 그 값에
f라는 활성화 함수를 적용하는 구조다.  
이러한 Dense층을 구성하는 기본적인 방법은 가중치인 W와 b를 각각 변수로
선언한 후 행렬곱을 통해 구하는 방법이다.

``` python
W = tf.Variable(tf.random.uniform([5,10],-1.0,-1.0)) # 가중치 행렬 선언
b = tf.Variable(tf.zeros([10])) # b 선언
y = tf.matmul(W,x) + b # 행렬곱 후 상수벡터 더함
```

하지만 텐서플로의 Dense를 이용하면 코드를 효율적으로 작성할 수 있다.
`dense=tf.keras.layers.Dense(...)`로 우선 객체를 생성해야 한다.

``` python
# 1. 객체 생성 후 다시 호출하면서 입력값 설정
dense = tf.keras.layers.Dense(...)
output = dense(input)

# 2. 객체 생성 시 입력값 설정
output = tf.keras.layers.Dense(...)(input)
```

활성화함수로 시그모이드 함수를 사용하고, 출력 값으로 10개의 값을
출력하는 완전 연결 계층을 정의해보자.

``` python
INPUT_SIZE = (20,1)
inputs = tf.keras.layers.Input(shape = INPUT_SIZE)
output = tf.keras.layers.Dense(units=10, activation=tf.nn.sigmoid)(inputs)
```

10개의 노드를 가지는 은닉층이 있고, 최종 출력 값은 2개의 노드가 있는
신경망 구조를 정의해보자.

``` python
INPUT_SIZE = (20,1)
inputs = tf.keras.layers.Input(shape = INPUT_SIZE)
hidden = tf.keras.layers.Dense(units=10, activation=tf.nn.sigmoid)(inputs)
output = tf.keras.layers.Dense(units=2, activation=tf.nn.sigmoid)(hidden)
```

> ## **tf.keras.layers.Dropout**

신경망 모델을 만들 때 생기는 여러 문제점 중 대표적 문제인 과적합은
정규화 방법을 통해서 해결하는데, 그 중 가장 대표적인 방법이
`dropout`이다. `tf.keras.layers.Dropout( ... )` 모듈을 사용하면 된다.
dropout을 적용할 때 설정할 수 있는 인자 값엔 *rate*(드랍아웃을 적용할
확률 지정), *noise\_shape*(정수형의 one dimension tensor값을 받음. 이
값을 지정함으로써 특정 값만 드랍아웃을 적용할 수 있음), *seed* 가 있다.

dropout을 적용한 신경망 구조를 정의해보자.

``` python
INPUT_SIZE = (20,1)
inputs = tf.keras.layers.Input(shape = INPUT_SIZE)
dropout = tf.keras.layers.Dropout(rate=0.2)(inputs)
hidden = tf.keras.layers.Dense(units=10, activation=tf.nn.sigmoid)(dropout)
output = tf.keras.layers.Dense(units=2, activation=tf.nn.sigmoid)(hidden)
```

dropout을 적용하려는 층의 노드를 객체에 적용하면 된다. 여러 층에
범용적으로 사용되어 과적합을 방지하기 때문에 모델을 구현할 때 자주
쓰인다.

> ## **tf.keras.layers.Conv1D**

합성곱(Convolution) 연산 중 `Conv1D`에 대해 알아보자. 합성곱 연산은
`Conv1D`, `Conv2D`, `Conv3D`로 나눠지는데 이 세개의 차이점은 합성곱의
방향과 출력값을 기준으로 나눠진다.

<table>
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:center;">
합성곱의 방향
</th>
<th style="text-align:center;">
출력값
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Conv1D
</td>
<td style="text-align:center;">
한 방향(가로)
</td>
<td style="text-align:center;">
1-D array(vector)
</td>
</tr>
<tr>
<td style="text-align:left;">
Conv2D
</td>
<td style="text-align:center;">
두 방향(가로,세로)
</td>
<td style="text-align:center;">
2-D array(matrix)
</td>
</tr>
<tr>
<td style="text-align:left;">
Conv3D
</td>
<td style="text-align:center;">
세 방향(가로,세로,높이)
</td>
<td style="text-align:center;">
3-D array(tensor)
</td>
</tr>
</tbody>
</table>
