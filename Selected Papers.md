# Notes for Selective Papers

## Cognitive Neuroscience

### *A map of object space in primate inferotemporal cortex*

- Nature 2020
- Pinglei Bao, Liang She, Mason McGill, Doris Y Tsao

**Mapping cognition to neural activation, not the reversed**

Tsao's paper on the functional mapping of lateral visual stream is ground-breaking, in the sense that, rather than mapping different types of visual stimuli to different cortical areas, they classified the visual stimuli according to the activation of different cortical area, i.e. they randomly generate stimuli for viewing, and conclude the shared property of the stimuli activating the same area.

This is somehow related to the notion of factor analysis, or the graph-modeling method in psychiatry. That is, we can find out the shared features in seemingly unrelated cognitive functions. This is particularly important when these shared features are supported by a same neural mechanism, which means that our theory- or intuition-driven classification of cognitive processes is indeed wrong.

For example, the most important contribution of this paper is that it shows how "high-level" cognitive and neural processes may actually be resolved into "low-level", statistic-driven computations. We may hypothesize that there is some cortical area responsible for face processing, object processing or so, which is actually based on our intuition, constraint by our language. But Tsao demonstrates that, from a data-driven perspective, the distinction is only about ratio. This is also very closely related to computer vision, where the extracted feature can be quite weird according to our intuition. But what if our visual system is indeed operating on, and organized by, these statistic-driven features, rather than something abstract like "concepts"?

## Computational Neuroscience

### *Re-visiting the echo-state property*

- *Neural Networks*, 2012
- Izzet B.Yildiz, Herbert Jaeger, Stefan J.Kiebel
- Follow-up on (Jaeger, 2001), fundamentals of reservoir computing

**The network dynamics**

- Standard network:
  $$
  x_{k + 1} = f(W^{ff}u_{k + 1} +Wx_k + W^{fb}y_k) \\
  y_k = g(W^{fo}[x_k;u_k])\\
  f(a) = \tanh (a), g(a) = \tanh (a) \text{ or } a
  $$

- Leaky network:
  $$
  \dot x = \frac{1}{\tau}(-ax + f(W^{ff}_u + Wx + W^{fb}y)) \\
  y = g(W^{fo}[x;u])\\
  f(a) = \tanh (a), g(a) = \tanh (a) \text{ or } a
  $$
  In simulation, let $$\Delta t = \frac{\delta_t}{\tau}$$, where $$\delta_t$$ is the simulation time step:
  $$
  x_{k + 1} = (1 - a\Delta t)x_k + \Delta tf(W^{ff}u_{k + 1} +Wx_k + W^{fb}y_k) \\
  y_k = g(W^{fo}[x_k;u_k])
  $$

- Note: Theorectical analysis of echo-state network are restricted to those without feedback ($$W_{fb} = 0$$)

**Echo-state Property**

- Roughly speaking, the ESP is a condition of asymptotic state convergence of the reservoir network, under the influence of driving input.

- A network $$F:X\times U \to X$$ (with the compactness condition) has the echo state property with respect to $$U$$: if for any left infinite input sequence $$u^{-\infin}\in U^{-\infin}$$ and any two state vector sequences $$x^{-\infin}_a, x_b^{-\infin}$$ that are both compatible with $$u^{-\infin}$$, it holds that $$x_a[0] = x_b[0]$$.

- In other words, there exists an "echo function" $$E$$ so that $$x[n] = E(\dots, u[n-1], u[n])$$, meaning that the network state is determined only by the input rather than initialization after enough time.

  - In fact, echo-state networks has "state-forgetiveness" and "input-forgetiveness", meaning that both the previous states and inputs will not influence the following states after enough time.

  - The presence of echo function allows the training of output weights to be considered as a simple linear regression minimizing the norm of this loss:
    $$
    \epsilon[n] = (f^{out})^{-1}y_{teach}[n] - \sum_{i = 1, 2, \dots, n}w_ie_i(\dots, u[n-1], u[n])
    $$
    where $$f^{out}$$ is the output activation and $$e_i$$ is the echo function for hidden neuron i, $$y_{teach}$$ is the expected output.

- Therefore, in order to get a better estimation of weights $$w_i$$, we want the independent variables $$e_i$$ to have a wide range.

  - Therefore, the network should provide a rich **"reservoir"** of dynamics to be "tapped" by the output weights.
  - This is achieved by using sparse and random connection, which keeps the subnetworks relatively decoupled.

**Conditions for Echo-state Property**

- Necessary but not sufficient condition (widely used): scale the recurrent weights to control the spectral radius
  - For standard model: scale recurrent weights $$W$$ to make $$\rho(W)$$ slightly smaller than 1
  - For leaky model: scale recurrent weights $$W$$ to make *effective spectral radius* $$\rho(\tilde W)$$ slightly smaller than 1, where $$\tilde W = (1 - a\Delta t)I_n + \Delta tW$$ is equivalent to the $$W$$ for standard model
- Sufficient condition:
  1. Start with a random, **non-negative** $$W$$
  2. Scale $$W$$ to make $$\rho(W)$$ (for standard model) or $$\rho(\tilde W)$$ (for leaky model) slightly smaller than 1
  3. Change the signs of a desired number of connections to negative (e.g., half of positive ones)

