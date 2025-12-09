# 중심 방정식과 블로흐 진동의 엄밀한 유도

**저자:** POSTECH 물리학과 이준석  
**날짜:** 2025년 12월 11일

## 개요 (Abstract)
본 문서는 주기적 포텐셜 하에서의 전자 거동을 기술하는 중심 방정식(Central Equation)을 기초부터 유도하고, 이를 통해 밴드 갭(Band Gap)과 코사인 형태의 에너지 밴드를 도출한다. 이후 외부 전기장이 인가된 상황에서 전자가 공간적으로 진동하는 블로흐 진동(Bloch Oscillation) 현상을 수식적으로 엄밀하게 기술한다.

---

## 1. 중심 방정식 (The Central Equation)

### 1.1 기초 설정: 푸리에 전개
주기 $a$를 갖는 1차원 결정 격자를 고려한다. 역격자 벡터는 $G = \frac{2\pi n}{a}$ ($n \in \mathbb{Z}$)이다.

1.  **퍼텐셜 에너지의 전개:**
    주기적 포텐셜 $V(x) = V(x+a)$는 푸리에 급수로 표현된다.
    $$
    V(x) = \sum_{G} V_G e^{iGx}, \quad V_G = \frac{1}{a} \int_0^a V(x) e^{-iGx} dx, \quad G = \frac{2\pi n}{a}
    $$
    여기서 $V(x)$는 실수이므로 $V_{-G} = V_G^*$ 조건을 만족한다.

2.  **파동함수의 전개 (블로흐 정리):**
    블로흐 정리에 의해 파동함수는 $\psi_k(x) = e^{ikx} u_k(x)$ 꼴을 가지며, 주기 함수 $u_k(x)$를 푸리에 전개하면 다음과 같다.
    $$
    \psi_k(x) = e^{ikx} u_k(x) \space \space \text{(Bloch Theorem)}, \quad \psi_k(x) = \sum_{G} C_{k-G} e^{i(k-G)x}
    $$
    이는 운동량 $k$인 상태가 격자($G$)와 상호작용하여 $k-G$ 상태들과 중첩됨을 의미한다.

### 1.2 슈뢰딩거 방정식 대입 및 인덱스 정리
시간 무관 슈뢰딩거 방정식에 위 식들을 대입한다.
$$
\left[ -\frac{\hbar^2}{2m} \frac{d^2}{dx^2} + \sum_{G'} V_{G'} e^{iG'x} \right] \sum_{G} C_{k-G} e^{i(k-G)x} = E \sum_{G} C_{k-G} e^{i(k-G)x}
$$

운동 에너지 항을 미분하고, 퍼텐셜 항의 이중 시그마를 정리한다.
$$
\sum_{G} \frac{\hbar^2 (k-G)^2}{2m} C_{k-G} e^{i(k-G)x} + \sum_{G'} \sum_{G} V_{G'} C_{k-G} e^{i(k-G+G')x} = E \sum_{G} C_{k-G} e^{i(k-G)x}
$$

**[인덱스 정리 과정]**
퍼텐셜 항의 지수 $k-G+G'$를 다른 항들과 비교하기 위해 $k-G''$ 형태로 맞춘다.
$G'' = G - G'$로 치환하면, $G' = G - G''$가 된다. 고정된 $G$에 대해 모든 $G'$를 더하는 것은 모든 $G''$를 더하는 것과 같다.
$$
\text{Potential Term} = \sum_{G''} \left( \sum_{G} V_{G-G''} C_{k-G} \right) e^{i(k-G'')x}
$$
이제 모든 항이 $e^{i(k-K)x}$ 형태(여기서 $K$는 일반적인 역격자 벡터)를 가지므로, 각 푸리에 성분의 계수를 비교하여 0이 되어야 함을 이용한다. $K$를 다시 $G$로, $G''$를 $G$로, $G$를 $G'$로 표기만 바꾸어 정리하면 최종적인 **중심 방정식**을 얻는다.

$$
(\lambda_{k-G} - E) C_{k-G} + \sum_{G'} V_{G-G'} C_{k-G'} = 0, \quad \lambda_k \equiv \frac{\hbar^2 k^2}{2m}
$$
여기서 $\lambda_k \equiv \frac{\hbar^2 k^2}{2m}$ 은 자유 입자의 운동 에너지이다.

---

## 2. 밴드 갭과 에너지 분산 관계

### 2.1 2-Band 근사 (Near Zone Boundary)
브릴루앙 영역 경계인 $k \approx \frac{\pi}{a}$ 근처에서는 운동 에너지 $\lambda_k$와 $\lambda_{k-G}$ (단, $G=\frac{2\pi}{a}$)가 거의 같아진다. 이 두 성분($C_k$와 $C_{k-G}$)이 지배적이므로 나머지 계수는 무시한다.
식은 두 개의 연립 방정식으로 축소된다.

$$
\text{$\lambda_k$ and $\lambda_{k-G}$ terms ($G=\frac{2\pi}{a}$)} = \begin{cases}
(\lambda_k - E) C_k + V_G C_{k-G} = 0 \\
V_{-G} C_k + (\lambda_{k-G} - E) C_{k-G} = 0
\end{cases}
$$

계수 행렬의 행렬식(Determinant)이 0이 되어야 자명하지 않은 해를 가진다. ($V_{-G}=V_G^*$ 이고 편의상 $V_G$를 실수 $U$로 둔다.)

$$
\begin{vmatrix} \lambda_k - E & U \\ U & \lambda_{k-G} - E \end{vmatrix} = (\lambda_k - E)(\lambda_{k-G} - E) - U^2 = 0
$$

### 2.2 에너지 해와 코사인 근사
위 2차 방정식을 $E$에 대해 풀면:
$$
E(k) = \frac{\lambda_k + \lambda_{k-G}}{2} \pm \sqrt{ \left( \frac{\lambda_k - \lambda_{k-G}}{2} \right)^2 + U^2 }
$$
$k = \pi/a$ (즉, $k = G/2$) 대입 시, $\lambda_k = \lambda_{k-G}$가 되어 제곱근 안이 $U^2$만 남는다.
$$
E_{\pm} = \lambda_{\pi/a} \pm |U|
$$
즉, 경계에서 $E_g = 2|U|$ 만큼의 에너지 갭이 열린다.

가장 낮은 밴드($-$ 해)는 $k=0$에서 극소, $k=\pi/a$에서 극대를 가지며 경계에서 기울기가 0이 된다. 이를 만족하는 가장 단순한 주기 함수인 코사인 함수로 근사한다. (Tight-binding 결과와 일치)
$$
E(k) \approx E_0 - \frac{\Delta}{2} \cos(ka)
$$
여기서 $\Delta$는 밴드 폭(Bandwidth)이다.

---

## 3. 블로흐 진동 (Bloch Oscillation)

### 3.1 준고전적 운동 방정식
전자(-e)에 일정한 전기장 $\mathcal{E}$가 인가될 때, 외력 $F = -e\mathcal{E}$는 결정 운동량 $k$의 시간 변화율과 같다.
$$
F = \hbar \frac{dk}{dt} = -e\mathcal{E} \quad \quad \quad \text{semiclassical approach}
$$
초기 조건 $k(0)=0$이라 가정하고 적분하면:
$$
k(t) = -\frac{e\mathcal{E}}{\hbar} t
$$
즉, $k$공간에서 전자는 등속 운동을 한다.

### 3.2 속도와 위치의 시간 진화
전자의 실제 이동 속도인 군속도(Group Velocity)는 다음과 같다.
$$
v_g(k) = \frac{1}{\hbar} \frac{dE}{dk}
$$
코사인 밴드 식을 대입하면:
$$
v_g(k) = \frac{1}{\hbar} \frac{d}{dk} \left( E_0 - \frac{\Delta}{2} \cos(ka) \right) = \frac{\Delta a}{2\hbar} \sin(ka)
$$
여기에 시간 의존성 $k(t)$를 대입한다.
$$
v_g(t) = \frac{\Delta a}{2\hbar} \sin\left( -\frac{e\mathcal{E}a}{\hbar} t \right) = -\frac{\Delta a}{2\hbar} \sin(\omega_B t)
$$
여기서 블로흐 주파수(Bloch Frequency) $\omega_B$를 정의하였다.
$$
\omega_B \equiv \frac{e\mathcal{E}a}{\hbar}
$$

마지막으로 속도를 적분하여 실공간 위치 $x(t)$를 구한다. ($x(0)=0$)
$$
\begin{align}
x(t) &= \int_0^t v_g(t') dt' = -\frac{\Delta a}{2\hbar} \int_0^t \sin(\omega_B t') dt' \\
&= -\frac{\Delta a}{2\hbar} \left[ -\frac{1}{\omega_B} \cos(\omega_B t') \right]_0^t \\
&= \frac{\Delta a}{2\hbar \omega_B} (\cos(\omega_B t) - 1)
\end{align}
$$
$\omega_B$를 대입하여 정리하면 최종 진동 식을 얻는다.
$$
x(t) = \frac{\Delta}{2e\mathcal{E}} (\cos(\omega_B t) - 1)
$$

---

## 4. 제너 터널링 (Zener Tunneling)

### 4.1 블로흐 진동의 한계와 밴드 간 전이
앞서 유도한 블로흐 진동은 전자가 가속되더라도 **단일 밴드(Single Band)** 내에 머무른다는 가정(단열 근사, Adiabatic approximation) 하에 성립한다. 하지만 외부 전기장 $\mathcal{E}$가 매우 강력해지면, 전자가 운동량 공간의 경계($k=\pi/a$)에서 반사되지 않고, 금지된 밴드 갭(Band Gap)을 뚫고 상위 밴드로 전이하는 현상이 발생한다. 이를 **제너 터널링**이라 한다.

### 4.2 물리적 메커니즘
실공간(Real Space)에서 강한 전기장은 에너지 밴드 전체를 기울어뜨린다.
전자의 위치 에너지 $V(x) = -e\mathcal{E}x$ 에 의해, 공간상에서 $x$가 변함에 따라 밴드 갭은 마치 **기울어진 포텐셜 장벽**처럼 작용한다.

전자가 터널링을 통해 건너가야 하는 '금지된 영역'의 공간적 폭 $w$는 다음과 같이 근사할 수 있다.
$$
w \approx \frac{E_g}{e\mathcal{E}} \quad \quad \text{Zener Tunneling Bandwidth}
$$
전기장 $\mathcal{E}$가 클수록 장벽의 폭 $w$가 좁아져 터널링 확률이 급격히 증가한다.

### 4.3 터널링 확률 (Landau-Zener Formula)
란다우-제너 공식(Landau-Zener Formula)에 따르면, 단위 시간당 전자가 밴드 갭을 넘어갈 터널링 확률 $P_{Zener}$는 다음과 같은 지수 함수 형태로 주어진다.

$$
P_{Zener} \propto \exp\left( - \frac{\pi^2}{h} \frac{E_g^2}{e \mathcal{E} a \Delta} \right)
$$

여기서 $E_g$는 밴드 갭, $\Delta$는 밴드 폭, $a$는 격자 상수이다.
* **약한 전기장 ($\mathcal{E} \ll E_g/ea$):** $P_{Zener} \to 0$. 전자는 밴드 내에 갇혀 **블로흐 진동**을 한다.
* **강한 전기장 ($\mathcal{E} \sim E_g/ea$):** $P_{Zener}$가 유의미해진다. 전자는 상위 밴드로 탈출하며, 이는 반도체 소자에서 **제너 항복(Zener Breakdown)**의 원인이 된다.

---

## 5. 결론
중심 방정식을 통해 주기적 포텐셜 내에서 에너지 밴드와 갭이 형성됨을 유도하였다. 이 밴드 구조($E \sim \cos k$)와 외부 전기장이 결합하면, 전자는 가속되어 무한히 빨라지는 것이 아니라 좁은 공간 영역을 왕복하는 블로흐 진동을 하게 된다. 이는 결정 운동량 $k$가 브릴루앙 영역 경계에서 브래그 반사를 겪기 때문이다.