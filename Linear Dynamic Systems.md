# Linear Dynamic Systems

## Introduction

- Instructor: Bruno Sinopoli, Tuesdays 4-5:30PM Green Hall 1100 or by appointment

- TA: Jonathan Gornet, Thursdays 5:30-7PM Green Hall 1121 or by appointment

- Recitation: Friday 11-12:30 PM, voluntary

- Contents:
  - Linear algebra review
  - Matrix theory
  - Linear systems - algebraic aspects
  - Linear systems - feedback aspects
  - Linear quadratic optimal control
  - Consensus Algorithm
- Textbook: C.T. Chen, Linear Systems, Theory and Design, 4th edition
- Reference book: J. Bay, Fundamentals of Linear State Space Systems, McGraw-Hill

## Description of a System

- Properties:

  - Causal: the output only depends on the current input and the historic inputs
  - Strictly causal: the output only depends on the historic inputs
  - Memoryless: the output only depends on the current input
  - Lumped or distributed: see below

- Three ways of modeling:

  - White box modeling: starting from first principles
  - Grey box: starting from physical laws but with some parameters to identify
  - Black box: no knowledge about the structure of the system
    - System identification: postulate a model and determine the coefficients with I/O experiments
  
- Differential equations:

  - I/O model (time domain):
    $$
    \sum_{i = 0}^n a_iy^{(i)}(t) = \sum_{j = 0}^m b_ju^{(j)}(t)
    $$
    where superscript indicates derivative.

    - The relative size of $n$ and $m$ determines the causality of the system:
      - $n > m$, strictly causal
      - $n = m$, causal,
      - $n < m$, non-causal
    - Assuming $m = 0, a^n = 1$ below for simplicity

  - Laplace transform (frequency domain):
    $$
    Y(s) = \int_0^\infin y(t)e^{-st}dt \\
    y(t) = \frac{1}{2\pi i}\lim_{T\to\infin}\int_{\gamma - iT}^{\gamma + iT}e^{st}F(s)ds \\
    $$
    where $\gamma$ is a real number related to the region of convergence for $Y(s)$. We have such property that:
    $$
    \mathcal{L}(f(t)) \triangleq F(s) \\
    \mathcal{L}(f^{(n)}(t)) = s^nF(s) - \sum_{k = 1}^ns^{n - k}f^{(k - 1)}(0^+)
    $$
    For LTI system, all derivatives at time 0 is 0, so we have:
    $$
    \sum_{i = 0}^n a_is^iY(s) = b_0U(s) \\
    Y(s) = H(s)U(s)
    $$
    where $H(s) = \frac{b_0}{\sum_{i = 0}^n a_is^i}$ is the transfer function. Besides:
    $$
    \mathcal{L}^{-1}(H(s)) \triangleq h(t) \\
    y(t) = h(t) \otimes u(t)
    $$
    where $h(t)$ is the impulse response and $\otimes$ denotes convolution.

  - State-space model:

    If we define the "state" of the system as:
    $$
    x_1(t) = y(t)\\
    x_2(t) = \dot{y}(t)\\
    \dots\\
    x_n(t) = y^{(n)}(t)
    $$
    Then
    $$
    \begin{pmatrix}\dot{x_1}\\ \dot{x_2}\\ \vdots\\ \dot{x_{n-1}}\\ \dot{x_n} \end{pmatrix} =
    \begin{pmatrix}
    0 & 1 & 0 & \dots & 0 \\
    0 & 0 & 1 & \dots & 0 \\
    \vdots & \vdots & \vdots & \vdots & \vdots \\
    0 & 0 & 0 & \dots & 1 \\
    -a_0 & -a_1 & -a_2 & \dots & -a_{n - 1}
    \end{pmatrix}
    \begin{pmatrix}x_1\\ x_2\\ \vdots\\ x_{n-1}\\ x_n \end{pmatrix} +
    \begin{pmatrix}0\\ 0\\ \vdots\\ 0\\ -b_0 \end{pmatrix}u(t) \\
    $$
    and $y(t) = \begin{pmatrix}1 & 0 & \dots & 0\end{pmatrix}\begin{pmatrix}x_1 & x_2 & \dots & x_n \end{pmatrix}^T$.

    However, this is only one in infinite many of ways that we can convert an I/O description into a state space description.

- State-space model (Internal model):

  - States:

    - The state $x(t_0)$ of a system at time $t_0$ is the information at $t_0$ that, together with the input $u(t), t \ge t_0$, uniquely determines the output $y(t)$ for all $t > t_0$.
    - If the state of a system is finite dimensional, the system is called lumped, otherwise it's called distributed.

  - In general, we have:
    $$
    \begin{cases}
    \dot{x} = Ax + Bu \\
    y = Cy + Du
    \end{cases}
    $$

    - $x\in \R^n$ is the state. $y\in \R^p$ is the output. $u\in\R^m$ is the input.
    - $A$ is the dynamics matrix. $B$ is the input matrix. $C$ is the output matrix. $D$ is the feedthrough matrix
