Deep learning with R
================



# 소개

이 파일은 슬기로운 통계생활님의 `딥러닝 공략집 with R`을 공부하며
review한 파일입니다.

## load packages

``` r
library(tidyverse)
library(torch)
library(dplyr)
```

## 텐서란?

텐서는 matrix 개념의 확장으로, 다차원 행렬이다. R의 Array와 비슷하다 GPU
계산에도 쓸 수 있다.

### 빈 텐서 만들기

속이 빈 텐서를 선언할 땐 `torch_empty(...)` 함수를 사용하여 만들 수
있다. 안에 들어있는 값은 별다른 의미를 갖지 않는 0근처의 값이다. 함수의
formals(list of arguments)의 첫 인자가 dot-dot-dot 값이 사용된 것으로
보아 임의의 n차원의 텐서를 허용하는 것으로 확인할 수 있다.

``` r
x <- torch_empty(5,3,3)
x
```

    ## torch_tensor
    ## (1,.,.) = 
    ##   0  0  0
    ##   0  0  0
    ##   0  0  0
    ## 
    ## (2,.,.) = 
    ##   0  0  0
    ##   0  0  0
    ##   0  0  0
    ## 
    ## (3,.,.) = 
    ##   0  0  0
    ##   0  0  0
    ##   0  0  0
    ## 
    ## (4,.,.) = 
    ##   0  0  0
    ##   0  0  0
    ##   0  0  0
    ## 
    ## (5,.,.) = 
    ##   0  0  0
    ##   0  0  0
    ##   0  0  0
    ## [ CPUFloatType{5,3,3} ]

``` r
# 텐서 x의 크기 확인
dim(x)
```

    ## [1] 5 3 3

`dim`함수로 텐서의 dimension을 확인 가능하며, *CPUFloatType{5,3}* 을
보아 CPU에서 접근 가능하며 실수 타입의 5x3x3 의 텐서가 선언됐음을 확인할
수 있다.

### 랜덤 텐서 만들기

텐서의 각 원소를 0에서 1 사이의 난수로 채워서 만든다. `torch_rand(...)`
함수를 사용한다. 이 역시도 첫 arguments가 dot-dot-dot이다.

``` r
formals(torch_rand)
```

    ## $...
    ## 
    ## 
    ## $names
    ## NULL
    ## 
    ## $dtype
    ## NULL
    ## 
    ## $layout
    ## torch_strided()
    ## 
    ## $device
    ## NULL
    ## 
    ## $requires_grad
    ## [1] FALSE

``` r
rand_tensor <- torch_rand(5,3)
rand_tensor
```

    ## torch_tensor
    ##  0.1025  0.1523  0.5977
    ##  0.5871  0.5873  0.1538
    ##  0.6106  0.2034  0.9081
    ##  0.9622  0.4385  0.8175
    ##  0.2705  0.1897  0.3697
    ## [ CPUFloatType{5,3} ]

텐서에 접근할 때는 R에서의 array의 접근 문법, subsetting 등으로
사용가능하다. 파이썬과 다르게, Rtorch이므로 index가 1부터 시작함을
확인할 수 있다.

``` r
rand_tensor[1,]
```

    ## torch_tensor
    ##  0.1025
    ##  0.1523
    ##  0.5977
    ## [ CPUFloatType{3} ]

``` r
rand_tensor[1:3,]
```

    ## torch_tensor
    ##  0.1025  0.1523  0.5977
    ##  0.5871  0.5873  0.1538
    ##  0.6106  0.2034  0.9081
    ## [ CPUFloatType{3,3} ]

``` r
rand_tensor[-1,]
```

    ## torch_tensor
    ##  0.2705
    ##  0.1897
    ##  0.3697
    ## [ CPUFloatType{3} ]

### 단위 텐서

4x4의 단위 텐서(identity matrix) 선언하는 방법은 다음과 같다.

``` r
x <- torch_eye(4)
x
```

    ## torch_tensor
    ##  1  0  0  0
    ##  0  1  0  0
    ##  0  0  1  0
    ##  0  0  0  1
    ## [ CPUFloatType{4,4} ]

### 영(0)텐서

모든 원소가 0으로 이루어진 텐서를 선언하는 것은 `torch_zeros(...)`함수를
사용할 수 있다.

``` r
x <- torch_zeros(3,5)
x
```

    ## torch_tensor
    ##  0  0  0  0  0
    ##  0  0  0  0  0
    ##  0  0  0  0  0
    ## [ CPUFloatType{3,5} ]

------------------------------------------------------------------------

지금까지는 미리 정해진 값이나 난수를 이용해서 하는 방법들을 확인해봤다.
이번 섹션에서는 직접 값을 정의해서 텐서를 만드는 방법을 알아본다.

### 텐서 직접 선언

``` r
y <- torch_tensor(matrix(c(1,2,3,4,5,6), ncol=2))
y
```

    ## torch_tensor
    ##  1  4
    ##  2  5
    ##  3  6
    ## [ CPUFloatType{3,2} ]

``` r
y1 <- torch_tensor(matrix(seq(0.1,1,length.out=10), ncol=2))
y1
```

    ## torch_tensor
    ##  0.1000  0.6000
    ##  0.2000  0.7000
    ##  0.3000  0.8000
    ##  0.4000  0.9000
    ##  0.5000  1.0000
    ## [ CPUFloatType{5,2} ]

``` r
y2 <- torch_tensor(1:5 %>% diag())
y2
```

    ## torch_tensor
    ##  1  0  0  0  0
    ##  0  2  0  0  0
    ##  0  0  3  0  0
    ##  0  0  0  4  0
    ##  0  0  0  0  5
    ## [ CPULongType{5,5} ]

결론은 R의 matrix선언 방법을 이용해 원하는 텐서를 선언한 후,
`torch_tensor()` 함수를 통해 텐서로 변환해주면 되는 것이다.

지금까지의 텐서 선언방법, 접근방법들을 살펴보면 결국 텐서와 R의 array는
같은 것으로 보일 수 있으나 둘의 객체 타입은 같지 않다. 간단한 연산을
통해 이를 확인해보자.

``` r
x <- torch_zeros(3,5)
x %*% t(x)
```

    ## Error in t.default(x): 인자가 행렬을 가지지 않습니다

위 error처럼 x는 matrix가 아니라고 나온다. 즉 근본이 다른 객체임을
확인할 수 있다.

### 텐서(tensor) 연산

``` r
torch_manual_seed(2021)
```

``` r
A <- torch_tensor(1:6)
B <- torch_rand(2,3)
C <- torch_rand(2,3,2)
A;B;C
```

    ## torch_tensor
    ##  1
    ##  2
    ##  3
    ##  4
    ##  5
    ##  6
    ## [ CPULongType{6} ]

    ## torch_tensor
    ##  0.5134  0.7426  0.7159
    ##  0.5705  0.1653  0.0443
    ## [ CPUFloatType{2,3} ]

    ## torch_tensor
    ## (1,.,.) = 
    ##   0.9628  0.2943
    ##   0.0992  0.8096
    ##   0.0169  0.8222
    ## 
    ## (2,.,.) = 
    ##   0.1242  0.7489
    ##   0.3608  0.5131
    ##   0.2959  0.7834
    ## [ CPUFloatType{2,3,2} ]

type을 살펴보자.

``` r
A$dtype
```

    ## torch_Long

``` r
B$dtype
```

    ## torch_Float

텐서 A는 정수형만 선언돼있으므로 실수형으로 바꿔보자.

``` r
A <- A$to(dtype=torch_double())
A$dtype
```

    ## torch_Double

아직은 tensor의 shape들도 다르므로 shape을 맞춰주자. `view()`로 맞출 수
있다.

``` r
A <- A$view(c(2,3)) # 벡터로 입력해야 함.
A
```

    ## torch_tensor
    ##  1  2  3
    ##  4  5  6
    ## [ CPUDoubleType{2,3} ]

이제 연산이 가능하다.

``` r
A+B
```

    ## torch_tensor
    ##  1.5134  2.7426  3.7159
    ##  4.5705  5.1653  6.0443
    ## [ CPUDoubleType{2,3} ]

사실은 shape만 맞으면, dtype이 달라도 연산이 가능하지만, 명시적으로
우리는 변환을 한 후 하는 것을 습관들여야 한다.

상수와의 연산도 가능하다.

``` r
A+2
```

    ## torch_tensor
    ##  3  4  5
    ##  6  7  8
    ## [ CPUDoubleType{2,3} ]

``` r
B^2
```

    ## torch_tensor
    ##  0.2636  0.5514  0.5125
    ##  0.3254  0.0273  0.0020
    ## [ CPUFloatType{2,3} ]

``` r
A %/% 3
```

    ## torch_tensor
    ##  0  0  1
    ##  1  1  2
    ## [ CPUDoubleType{2,3} ]

``` r
A %% 3
```

    ## torch_tensor
    ##  1  2  0
    ##  1  2  0
    ## [ CPUDoubleType{2,3} ]

``` r
torch_sqrt(A); A^(1/2)
```

    ## torch_tensor
    ##  1.0000  1.4142  1.7321
    ##  2.0000  2.2361  2.4495
    ## [ CPUDoubleType{2,3} ]

    ## torch_tensor
    ##  1.0000  1.4142  1.7321
    ##  2.0000  2.2361  2.4495
    ## [ CPUDoubleType{2,3} ]

``` r
torch_log(B); log(B)
```

    ## torch_tensor
    ## -0.6667 -0.2977 -0.3342
    ## -0.5613 -1.8002 -3.1166
    ## [ CPUFloatType{2,3} ]

    ## torch_tensor
    ## -0.6667 -0.2977 -0.3342
    ## -0.5613 -1.8002 -3.1166
    ## [ CPUFloatType{2,3} ]

곱셉은 `torch_matmul()`이나 `torch_mm()`함수를 사용한다.

``` r
D <- C[1,,]
torch_matmul(B,D)
```

    ## torch_tensor
    ##  0.5800  1.3409
    ##  0.5664  0.3381
    ## [ CPUFloatType{2,2} ]

``` r
torch_mm(B,D)
```

    ## torch_tensor
    ##  0.5800  1.3409
    ##  0.5664  0.3381
    ## [ CPUFloatType{2,2} ]

``` r
B$mm(D)
```

    ## torch_tensor
    ##  0.5800  1.3409
    ##  0.5664  0.3381
    ## [ CPUFloatType{2,2} ]

``` r
B$matmul(D)
```

    ## torch_tensor
    ##  0.5800  1.3409
    ##  0.5664  0.3381
    ## [ CPUFloatType{2,2} ]

### 텐서의 전치

전치는 다음의 문법구조로 사용 가능하다.
`tensor_transpose(input,dim0,dim1)`. 다음의 예시들을 살펴보자.

``` r
torch_transpose(A,1,2)
```

    ## torch_tensor
    ##  1  4
    ##  2  5
    ##  3  6
    ## [ CPUDoubleType{3,2} ]

``` r
A$transpose(1,2)
```

    ## torch_tensor
    ##  1  4
    ##  2  5
    ##  3  6
    ## [ CPUDoubleType{3,2} ]

3차원 텐서의 경우를 살펴보자. 두번째 차원과 세번쨰 차원을 뒤집어야
면(depth)가 바뀌지 않으므로 다음과 같이 전치한다.

``` r
C
```

    ## torch_tensor
    ## (1,.,.) = 
    ##   0.9628  0.2943
    ##   0.0992  0.8096
    ##   0.0169  0.8222
    ## 
    ## (2,.,.) = 
    ##   0.1242  0.7489
    ##   0.3608  0.5131
    ##   0.2959  0.7834
    ## [ CPUFloatType{2,3,2} ]

``` r
torch_transpose(C,2,3)
```

    ## torch_tensor
    ## (1,.,.) = 
    ##   0.9628  0.0992  0.0169
    ##   0.2943  0.8096  0.8222
    ## 
    ## (2,.,.) = 
    ##   0.1242  0.3608  0.2959
    ##   0.7489  0.5131  0.7834
    ## [ CPUFloatType{2,2,3} ]

### 다차원 텐서와 1차원 벡터 텐서의 연산

R에서는 `recycling rule`이 있어서 모양이나 차원이 맞지 않는 객체들끼리의
연산 시 자동으로 맞춰줘서 연산을 해준다. Rtorch에서도 다음과 같은 기능이
제공된다. 자주 사용되지는 않을지라도 알아두자.

``` r
A
```

    ## torch_tensor
    ##  1  2  3
    ##  4  5  6
    ## [ CPUDoubleType{2,3} ]

``` r
A + torch_tensor(1:3)
```

    ## torch_tensor
    ##  2  4  6
    ##  5  7  9
    ## [ CPUDoubleType{2,3} ]

### 1차원 텐서끼리의 연산, 내적과 외적

``` r
A_1 <- A$view(c(1,-1)) # argument -1은 1행에 맞춰 자동으로 열을 맞춰주는 역할.
A_1
```

    ## torch_tensor
    ##  1  2  3  4  5  6
    ## [ CPUDoubleType{1,6} ]

``` r
A_2 <- A$view(c(-1,1))
A_2
```

    ## torch_tensor
    ##  1
    ##  2
    ##  3
    ##  4
    ##  5
    ##  6
    ## [ CPUDoubleType{6,1} ]

``` r
A_1$mm(A_2)
```

    ## torch_tensor
    ##  91
    ## [ CPUDoubleType{1,1} ]

``` r
A_2$mm(A_1)
```

    ## torch_tensor
    ##   1   2   3   4   5   6
    ##   2   4   6   8  10  12
    ##   3   6   9  12  15  18
    ##   4   8  12  16  20  24
    ##   5  10  15  20  25  30
    ##   6  12  18  24  30  36
    ## [ CPUDoubleType{6,6} ]

``` r
torch_matmul(A_1,A_2)
```

    ## torch_tensor
    ##  91
    ## [ CPUDoubleType{1,1} ]

두 행렬 객체가 있다고 가정해보자(a x b / c x d). 우리는 b와 c가 같아야
행렬의 곱이 가능하다는걸 알고 있기에 항상 연산 시 주의해야한다.

``` r
A_1 %>% dim()
```

    ## [1] 1 6

``` r
A_1$size()
```

    ## [1] 1 6

### 텐서의 이동, CPU and GPU

딥러닝에서는 네트워크 구조가 조금만 복잡해져도 연산의 양이
기하급수적으로 늘어난다. 때문에 GPU는 사실 필수적이다. `torch`패키지의
자료형에서는 현재 텐서가 어디에 저장되어 있는지가 표시된다.

``` r
a <- torch_tensor(1:4)
a
```

    ## torch_tensor
    ##  1
    ##  2
    ##  3
    ##  4
    ## [ CPULongType{4} ]

a가 현재 CPU의 메모리를 사용하고 있음을 알 수 있다.

### GPU 사용 가능 체크

CUDA나 cudNN, 드라이버 등의 설치를 통해 CUDA를 이용할 수 있는데, GPU의
접근성은 `cuda_is_available()`함수를 사용하면 된다.

``` r
cuda_is_available()
```

    ## [1] TRUE

### CPU to GPU

이미 정의된 텐서 a를 GPU로 옮길 땐 `cuda()`함수를 이용하면 된다.

``` r
a$cuda()
```

    ## torch_tensor
    ##  1
    ##  2
    ##  3
    ##  4
    ## [ CUDALongType{4} ]

``` r
a # cuda() function은 임시의 이동임을 알 수 있다.
```

    ## torch_tensor
    ##  1
    ##  2
    ##  3
    ##  4
    ## [ CPULongType{4} ]

``` r
gpu <- torch_device("cuda")
a$to(device=gpu) # 이것도 임시의 이동
```

    ## torch_tensor
    ##  1
    ##  2
    ##  3
    ##  4
    ## [ CUDALongType{4} ]

``` r
a
```

    ## torch_tensor
    ##  1
    ##  2
    ##  3
    ##  4
    ## [ CPULongType{4} ]

옮길 때 `dtype`을 명시하여 자료형도 바꿀 수 있다.

``` r
a$to(device=gpu,dtype=torch_double())
```

    ## torch_tensor
    ##  1
    ##  2
    ##  3
    ##  4
    ## [ CUDADoubleType{4} ]

### GPU to CPU

GPU 상에 직접 텐서를 만드는 방법은 다음과 같다

``` r
b <- torch_tensor(1:4, device=gpu)
b
```

    ## torch_tensor
    ##  1
    ##  2
    ##  3
    ##  4
    ## [ CUDALongType{4} ]

CPU로 옮길 땐 `cpu()`를 이용해서 가능하다.

``` r
b$cpu() # 이 역시도 임시적인 이동
```

    ## torch_tensor
    ##  1
    ##  2
    ##  3
    ##  4
    ## [ CPULongType{4} ]

``` r
b
```

    ## torch_tensor
    ##  1
    ##  2
    ##  3
    ##  4
    ## [ CUDALongType{4} ]

`to`함수를 이용하면,

``` r
cpu <- torch_device("cpu")
a$to(device=cpu) # 이 역시도 임시적임.
```

    ## torch_tensor
    ##  1
    ##  2
    ##  3
    ##  4
    ## [ CPULongType{4} ]

### R6(OOP)와 텐서

`torch`의 코드들을 보면 기존의 R의 사용법과는 약간 다른점들이 눈에
보인다. 이것의 근본적인 이유는 `torch`패키지가 바로 `R6 OOP`를 기반으로
하고 있기 때문이다. 자세히 말하면 `torch`의 텐서와 신경망들이 `R6`
패키지의 클래스들로 정의되어 있기 때문에, 일반적인 R 패키지들보다 `$`한
method 접근이 가능하다.

OOP, Object-Oriented System의 핵심은 클래스와 클래스에 따라 적용되는
방법(method)에 있다. 객체의 클래스는 객체에 적용되는 방법을 결정하며
다른 클래스와의 관계를 정의한다. 즉 특정 방법(함수)을 선택할 때 객체의
클래스 정보가 사용되며 클래스에 따라 동일한 함수를 적용할 때에도 다른
결과를 반환한다. 클래스는 계층구조를 갖고 있어서 만약 특정 클래스에
적용할 하위 방법(child)가 존재하지 않는다면 parent가 적용된다. *Hadely
Wickham*의 *Advanced R*의 OOP를 참고하면, 저자는 R의 OOP중 S3, R6, S4가
제일 중요하다고 생각하고 있으며, S3,S4는 base R에서 제공하며, R6는
패키지를 통해 제공되고 이는 RC(reference classes)와 비슷하다고 한다.  
R에서 사용되고 있는 객체지향 시스템에는 크게 네 가지가 있다. 우리는 이
중 `R6`를 먼저 보고자 한다.

-   S3 implement a style of OO programming called generic-function OO.
-   S4 works similary to S3, but is more formal.
-   Reference classes, called RC for short, are quite different from S3
    and S4, but quite similar to R6
-   base types, the internal C-level types that underlie the orher OO
    systems.

``` r
library(R6)
```

### 클래스(class)와 멤버함수(Method), 그리고 필드(Field)

`R6` 패키지에는 딱 하나의 함수가 존재한다. 바로 `R6Class()` 함수이다. 이
함수의 입력 값 중 눈여겨 볼 것은 두가지이다.

``` r
formals(R6Class)
```

    ## $classname
    ## NULL
    ## 
    ## $public
    ## list()
    ## 
    ## $private
    ## NULL
    ## 
    ## $active
    ## NULL
    ## 
    ## $inherit
    ## NULL
    ## 
    ## $lock_objects
    ## [1] TRUE
    ## 
    ## $class
    ## [1] TRUE
    ## 
    ## $portable
    ## [1] TRUE
    ## 
    ## $lock_class
    ## [1] FALSE
    ## 
    ## $cloneable
    ## [1] TRUE
    ## 
    ## $parent_env
    ## parent.frame()
    ## 
    ## $lock

첫번째는 클래스 이름 `classname`이고 두번째는 공개될 정보들을 담을
`public`이라는 입력값이다. `public`에는 우리가 만들 클래스에서 사용이
가능한 멤버함수들(methods)과 변수(fields)들을 넣은 리스트형태가
들어간다.

``` r
ExampleClass <- R6Class(classname = "Example", public = list(
    # 변수(fields) 정의
    # 멤버함수(methods) 정의
))
ExampleClass
```

    ## <Example> object generator
    ##   Public:
    ##     clone: function (deep = FALSE) 
    ##   Parent env: <environment: R_GlobalEnv>
    ##   Locked objects: TRUE
    ##   Locked class: FALSE
    ##   Portable: TRUE

이름을 정하는 방식은 암묵적인 규칙이 있다.  
1. 클래스의 이름은 `UpperCamelcase` 형식으로 짓는다. 1. 두번째 리스트에
들어가는 요소들의 이름은 `snake_case`를 사용한다.

### 클래스는 왜 필요할까?

클래스를 이해하는 데 아주 효과적인 예제 학생 클래스를 살펴보자.

### 학생자료 입력 예제

다음의 코드를 생각해보자.

``` r
student <- function(){
    list()
} # 빈 리스트를 반환하는 function 선언.

KIM <- student() # KIM 이라는 심볼의 리스트 생성
LEE <- student() # Lee 라는 심볼의 리스트 생성
```

``` r
KIM
```

    ## list()

``` r
LEE
```

    ## list()

student function을 이용하여 KIM과 LEE라는 학생의 정보를 담은 리스트를
위처럼 만들 수 있다. 이제 추가정보를 저장하려 한다고 가정해보자.

``` r
KIM$first <- "KIM"
KIM$last <- "MINWOO"
KIM$email <- "KIM-minwoo@gmail.com"
KIM$midterm <- 70
KIM$final <- 50

LEE$first <- "LEE"
LEE$last <- "Hyerin"
LEE$email <- "LEE-Hyerin@gmail.com"
LEE$midterm <- 65
LEE$final <- 80
```

``` r
KIM
```

    ## $first
    ## [1] "KIM"
    ## 
    ## $last
    ## [1] "MINWOO"
    ## 
    ## $email
    ## [1] "KIM-minwoo@gmail.com"
    ## 
    ## $midterm
    ## [1] 70
    ## 
    ## $final
    ## [1] 50

``` r
LEE
```

    ## $first
    ## [1] "LEE"
    ## 
    ## $last
    ## [1] "Hyerin"
    ## 
    ## $email
    ## [1] "LEE-Hyerin@gmail.com"
    ## 
    ## $midterm
    ## [1] 65
    ## 
    ## $final
    ## [1] 80

이런 코드는 OOP 관점이나 functional 관점에서 비효율적이다. 우리는
`R6Class()`를 사용하여 어떻게 줄일 수 있는지 알아보자.

### 클래스(Class) 정의하기

우리는 위의 예에서 `Student` 클래스를 미리 정의해놓을 수 있다.

``` r
Student <- R6Class("Student", list(
    # 필요한 변수 (field) 선언
    first = NULL,
    last = NULL,
    email = NULL,
    midterm = NA,
    final = NA,
    
    # 클래스 안의 객체를 만들때 사용되는 initialize
    initialize = function(first, last, midterm, final){
        self$first = first
        self$last  = last
        self$email = glue::glue("{tolower(first)}-{tolower(last)}@gmail.com")
        self$midterm = midterm
        self$final = final
    }    
))

Student
```

    ## <Student> object generator
    ##   Public:
    ##     first: NULL
    ##     last: NULL
    ##     email: NULL
    ##     midterm: NA
    ##     final: NA
    ##     initialize: function (first, last, midterm, final) 
    ##     clone: function (deep = FALSE) 
    ##   Parent env: <environment: R_GlobalEnv>
    ##   Locked objects: TRUE
    ##   Locked class: FALSE
    ##   Portable: TRUE

결과값을 유심히 살펴보면, `<Student> object generator` 라는 부분을 통해
`Student` 클래스는 객체를 만들어내는 생성자임을 알 수 있다.
`new()`함수를 사용하여 다음과 같이 만들 수 있다.

``` r
KIM <- Student$new("KIM","minwoo",70,50)
LEE <- Student$new("Lee","Hyerin",65,80)
```

``` r
KIM
```

    ## <Student>
    ##   Public:
    ##     clone: function (deep = FALSE) 
    ##     email: kim-minwoo@gmail.com
    ##     final: 50
    ##     first: KIM
    ##     initialize: function (first, last, midterm, final) 
    ##     last: minwoo
    ##     midterm: 70

``` r
LEE
```

    ## <Student>
    ##   Public:
    ##     clone: function (deep = FALSE) 
    ##     email: lee-hyerin@gmail.com
    ##     final: 80
    ##     first: Lee
    ##     initialize: function (first, last, midterm, final) 
    ##     last: Hyerin
    ##     midterm: 65

재사용이나 효율성 측면에서 훨씬 좋은 코드임을 알 수 있다.

### print()를 이용한 결과물 정리

정의된 클래스는 기본적으로 동작하는 함수들을 덮어서 쓸 수 있다.

``` r
Student <- R6Class("Student", list(
    # 필요한 변수 (field) 선언
    first = NULL,
    last = NULL,
    email = NULL,
    midterm = NA,
    final = NA,
    
    # 클래스 안의 객체를 만들때 사용되는 initialize
    initialize = function(first, last, midterm, final){
        self$first = first
        self$last  = last
        self$email = glue::glue("{tolower(first)}-{tolower(last)}@gmail.com")
        self$midterm = midterm
        self$final = final
    },
    print = function(...){
        cat("Student: \n")
        cat(glue::glue("
                Name  : {self$first} {self$last}
                E-mail: {self$email}
                Midterm Score : {self$midterm}
                Final Score: {self$final}
            "))
        invisible(self)
    }
))

soony <- Student$new("Soony", "Kim", 70, 20)
soony
```

    ## Student: 
    ##     Name  : Soony Kim
    ##     E-mail: soony-kim@gmail.com
    ##     Midterm Score : 70
    ##     Final Score: 20

### set을 이용한 클래스 조정

soony는 `print` 멤버 함수가 추가된 상태로 선언된 객체지만, KIM,LEE는
그렇지 않다. 이럴 때 클래스를 재정의 하는 것 보다 `set()` 을 이용해서
변수나 함수를 추가할 수 있다.

``` r
Student$set("public", "total", NA)
Student$set("public", "calculate_total", function(){
    self$total <- self$midterm + self$final
    invisible(self)
})
```

`invisible()`함수는 결과를 반환하되, 결과물을 보여주지 않는 것인데,
클래스에서 함수를 정의할 때 반드시 `invisible(self)`를 반환해줘야만
한다. 따라서 함수이지만 함수와는 다른 이 클래스 안의 함수들을 멤버함수
`method()`라고하여 일반 함수와 구분을 지어서 부른다.

``` r
jelly  <- Student$new("Jelly","Lee",35,23)
jelly
```

    ## Student: 
    ##     Name  : Jelly Lee
    ##     E-mail: jelly-lee@gmail.com
    ##     Midterm Score : 35
    ##     Final Score: 23

``` r
jelly$total
```

    ## [1] NA

``` r
jelly$calculate_total()
jelly$total
```

    ## [1] 58

### 상속(Inheritance) - 클래스 물려받기

이제까지 사용해온 학생 개념, `Student` 클래스를 좀 더 세분화를 해 학교
별로 나눠보자. `Student` 클래스를 상속받는 서브 클래스는 다음과 같이
생성할 수 있다.

``` r
UspStudent <- R6Class("UspStudent",
    inherit = Student, # 상속 결정
    public = list(
        university_name = "University of Statistics Playbook",
        class_year = NA,
        average = NA,
        calculate_average = function(){
            self$average <- mean(c(self$midterm, self$final))
            invisible(self)
        },
        calculate_total = function(){
            cat("The total score of midterm and final exam is calculated. \n")
            super$calculate_total()
        }
    )
)

sanghoon <- UspStudent$new("Sanghoon", "Park", 80, 56)
sanghoon
```

    ## Student: 
    ##     Name  : Sanghoon Park
    ##     E-mail: sanghoon-park@gmail.com
    ##     Midterm Score : 80
    ##     Final Score: 56

상위 클래스가 가지고 있는 calcualte\_total() 멤버함수에 접근하여, 새롭게
고쳐 사용하는 것도 가능하다.

``` r
sanghoon$calculate_average()
sanghoon$average
```

    ## [1] 68

``` r
sanghoon$calculate_total()
```

    ## The total score of midterm and final exam is calculated.

``` r
sanghoon$total
```

    ## [1] 136

### 공개 정보와 비공개 정보의 필요성

앞에서 살펴본 `R6Class()` 함수에서 클래스를 만들고 사용하다보면, 때로는
클래스 안의 함수들을 사용하기 위해서 만들어야하는 변수나 함수들이
있는데, 이들을 굳이 사용자들에게 보여줄 필요는 없으므로 public과
private들로 분류함으로써 조절 사용 가능하다.

``` r
UspStudent <- R6Class("UspStudent",
    inherit = Student,
    public = list(
        university_name = "University of Statistics Playbook",
        class_year = NA,
        calculate_average = function(){
            private$.average <- mean(c(self$midterm, self$final))
            cat("Average score is", private$.average)
            invisible(self)
        },
        calculate_total = function(){
            cat("The total score of midterm and final exam is calculated. \n")
            super$calculate_total()

        }
    ),
    private = list(
        .average = NA    
    )
)

taemo <- UspStudent$new("Taemo", "Bang", 80, 56)
taemo$calculate_average()
```

    ## Average score is 68

위의 `UspStudent` 클래스에는 비공개 정보 `average` 변수가 `private`에
감싸져서 입력 되었음을 주목하자.

## 순전파 (Forward propagation)

2단 순전파 신경망을 간단하게 구현해보자.  
우리가 사용할 데이터는 다음과 같다.

``` r
X = matrix(c(1,2,3,4,5,6), byrow=T, ncol=2);X
```

    ##      [,1] [,2]
    ## [1,]    1    2
    ## [2,]    3    4
    ## [3,]    5    6

먼저 첫번째 표본인 (1,2)가 어떤 경로로 신경망을 지나가게 될 지
생각해보자.  
먼저 (1,2)는 각각 입력층에 x1,x2로 들어가게 되고, b11,b21의 가중치를
곱해주게 되고, 그 다음 sigmoid 함수에 적용되게 된다. 이를 직접 코드로
구현해보면,

``` r
set.seed(1234)

X <- torch_tensor(matrix(1:2, ncol=2, byrow=T),
                  dtype = torch_double())
X
```

    ## torch_tensor
    ##  1  2
    ## [ CPUDoubleType{1,2} ]

``` r
# beta_1 = beta_11, beta_12
beta_1 <- torch_tensor(matrix(runif(2), ncol=1), dtype=torch_double())
beta_1
```

    ## torch_tensor
    ##  0.1137
    ##  0.6223
    ## [ CPUDoubleType{2,1} ]

``` r
z_21 <- X$mm(beta_1)
z_21
```

    ## torch_tensor
    ##  1.3583
    ## [ CPUDoubleType{1,1} ]

``` r
library(sigmoid)
```

    ## Warning: package 'sigmoid' was built under R version 4.0.5

``` r
a_21 <- sigmoid(z_21)
a_21
```

    ## torch_tensor
    ##  0.7955
    ## [ CPUDoubleType{1,1} ]

``` r
gamma_1 <- runif(1)

z_31 <- a_21 * gamma_1
z_31
```

    ## torch_tensor
    ##  0.4847
    ## [ CPUDoubleType{1,1} ]

``` r
y_hat <- sigmoid(z_31)
y_hat
```

    ## torch_tensor
    ##  0.6188
    ## [ CPUDoubleType{1,1} ]

회귀분석과 연결지어 이 과정을 생각해보면, 각각의 은닉층에서 시그모이드
함수를 통해 회귀분석 예측 결과를 모아놓고, 마지막 아웃풋에 예측값들을
모아 마지막 노드에서 합치며 조금 더 좋은 값을 만들어내는 것으로도 해석
가능하다.

우리는 방금 한 개의 표본이 은닉층 중 하나의 노드를 거쳐가는 과정을
구현해봤는데, 이번엔 한 개의 표본이 전체 은닉층을 지나가는 과정을
구현해보자.

``` r
X <- torch_tensor(matrix(1:2, byrow=T, ncol=2),
                  dtype = torch_double())
X
```

    ## torch_tensor
    ##  1  2
    ## [ CPUDoubleType{1,2} ]

``` r
beta_1 <- torch_tensor(matrix(runif(2), ncol=1),
                       dtype = torch_double())
beta_2 <- torch_tensor(matrix(runif(2), ncol=1),
                       dtype = torch_double())
beta_3 <- torch_tensor(matrix(runif(2), ncol=1),
                       dtype = torch_double())

beta_1;beta_2;beta_3
```

    ## torch_tensor
    ##  0.6234
    ##  0.8609
    ## [ CPUDoubleType{2,1} ]

    ## torch_tensor
    ##  0.6403
    ##  0.0095
    ## [ CPUDoubleType{2,1} ]

    ## torch_tensor
    ##  0.2326
    ##  0.6661
    ## [ CPUDoubleType{2,1} ]

``` r
beta <- torch_cat(c(beta_1,beta_2,beta_3),2)
beta
```

    ## torch_tensor
    ##  0.6234  0.6403  0.2326
    ##  0.8609  0.0095  0.6661
    ## [ CPUDoubleType{2,3} ]

``` r
z_2 <- X$mm(beta)
z_2
```

    ## torch_tensor
    ##  2.3452  0.6593  1.5647
    ## [ CPUDoubleType{1,3} ]

``` r
library(sigmoid)
a_2 <- sigmoid(z_2)
a_2
```

    ## torch_tensor
    ##  0.9126  0.6591  0.8270
    ## [ CPUDoubleType{1,3} ]

``` r
gamma_1 <- runif(1)
gamma_2 <- runif(1)
gamma_3 <- runif(1)
gamma <- torch_tensor(matrix(c(gamma_1,
                               gamma_2, 
                               gamma_3), ncol = 1),
                      dtype = torch_double())
z_3 <- a_2$mm(gamma)
z_3
```

    ## torch_tensor
    ##  1.3771
    ## [ CPUDoubleType{1,1} ]

``` r
y_hat <- sigmoid(z_3)
y_hat
```

    ## torch_tensor
    ##  0.7985
    ## [ CPUDoubleType{1,1} ]

이번엔 전체 표본들에 대해서 경로 전체를 생각해보자.

``` r
X <- torch_tensor(matrix(1:6, ncol = 2, byrow = T),
                  dtype = torch_double()) 
X
```

    ## torch_tensor
    ##  1  2
    ##  3  4
    ##  5  6
    ## [ CPUDoubleType{3,2} ]

``` r
beta <- torch_tensor(matrix(runif(6), ncol = 3),
                     dtype = torch_double())
beta
```

    ## torch_tensor
    ##  0.2827  0.2923  0.2862
    ##  0.9234  0.8373  0.2668
    ## [ CPUDoubleType{2,3} ]

``` r
z_2 <- X$mm(beta)
z_2
```

    ## torch_tensor
    ##  2.1296  1.9669  0.8199
    ##  4.5419  4.2261  1.9260
    ##  6.9543  6.4854  3.0320
    ## [ CPUDoubleType{3,3} ]

``` r
a_2 <- sigmoid(z_2)


gamma <- torch_tensor(matrix(runif(3), ncol = 1),
                      dtype = torch_double())


z_3 <- a_2$mm(gamma)
z_3
```

    ## torch_tensor
    ##  0.5904
    ##  0.6900
    ##  0.7205
    ## [ CPUDoubleType{3,1} ]

``` r
y_hat <- sigmoid(z_3)
y_hat
```

    ## torch_tensor
    ##  0.6435
    ##  0.6660
    ##  0.6727
    ## [ CPUDoubleType{3,1} ]

이 계산을 직접 해봄으로써 layer의 합, 곱 연산의 순서를 이해하는 것은
귀찮을지라도 후에 복잡한 작업의 순서를 이해하는 데 있어서 꼭 필요한
연습이라 생각된다.
