# Calculus

## Derivatives & Relevant Concepts

- Derivative (全导数):
  - the changing speed of f(X) when X travels along a curve $$l(t)$$
  - $$f'(t) = \lim_{h\to 0} \frac{f(X(t + h)) - f(X(t))}{h}, X \in \R^n$$
  - $$f'(t) = \nabla f \cdot (x_1', x_2', \dots, x_n')$$
- Partial derivative:
  - when $$l(t)$$ is exactly one axis
  - $$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, x_2, \dots, x_i + h, \dots, x_n) - f(x_1, x_2, \dots, x_i, \dots, x_n)}{h}$$
- Directional derivative:
  - when $$l(t)$$ is a straight line in one direction $$\bold{v}$$
  - $$\nabla_{\bold{V}}f(X) = \lim_{h \to 0}\frac{f(X + h\bold{v}) - f(X)}{h} = \nabla f \cdot \bold{v}$$
- Divergence:
  - Definition: 
    - The divergence of a vector field $$\bold{F}(\bold{x})$$ at a point x0 is defined as the limit of the ratio of the surface integral of F out of the surface of a closed volume V enclosing x0 to the volume of V
    - $$\nabla \cdot \bold{F} = \lim_{V \to 0}\frac{1}{|V|} \oiint_{S(V)}\bold{F}\cdot\bold{\vec{n}}\ dS$$, where *n* is the outward unit normal to *S*
    - Relationship with *Gauss's theorem*: $$\underbrace{\int\dots\int_U}_{n} \nabla\cdot\bold{F} dV = \underbrace{\int\dots\int_{\partial U}}_{n - 1}\bold{F}\cdot\bold{n}\ dS$$
  - Explanation:
    - The extent to which the point behaves like a source: sink if negative, no flux if zero, source if positive
  - Calculation: 
    - $$\nabla\cdot \bold{F} = (\frac{\partial}{\partial x_1}, \frac{\partial}{\partial x_2}, \dots, \frac{\partial}{\partial x_n}) \cdot \bold{F} = \sum_{j = 1}^{n}\frac{\partial F_{x_j}}{\partial x_j}$$
- Laplacian:
  - Definition: 
    - the Laplacian of a scalar function is the divergence of its gradient field
    - $$\Delta f = \nabla^2f = \nabla \cdot(\nabla f) = \sum_{i=1}^n\frac{\partial^2f}{\partial x^2_i}$$
  - Explanation:
    - If $$\phi$$ denotes the electrostatic potential associated to a charge distribution *q*, then $$q = -\epsilon_0\Delta\phi$$, where ε0 is the electric constant.
    - If *f* is the phase distribution, then its Lapalican can indicate where may be the source of a spherical travelling wave