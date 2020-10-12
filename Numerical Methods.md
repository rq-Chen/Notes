# Numerical Methods

## Ordinary Differential Equations

### Euler's Method

- *Euler's method* is an iterative method for first-order *ordinary differential equations (ODE)*, but actually all ODE can be reduced to first-order.
- It starts from one point on the curve, then walks a little forward on its tangent in each step: $$y_{n+1} = y_n + hf(t_n, y_n)$$, where $$f(t, y) = y'(t)$$.

## Interpolation

### Natural Neighbour Interpolation

- Interpolate a point *P* by the weighted sum of its neighbours {*S*} in space, where the weights are calculated according to the *Voronoi tessellation* with or without *P*.
- *Sibson weights*: weights can be the ovelapping volume of the old Voronoi cell of *S* (noted as cell S) and the new Voronoi cell of *P* (noted as cell P) divided by the latter's volume.
- *Laplace weights*: weights are the measure of the interface between cell P and new cell S divided by the distance between P and S, then normalized (See [wiki](https://en.wanweibaike.com/wiki-Natural%20neighbour%20interpolation)).