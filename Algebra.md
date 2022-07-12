# Algebra

## Singular Value Decomposition

- Linear transform view: decomposite any linear transform into three operations:
  - a rotation/reflection
  - followed by a coordinate scaling (with the singular values as ratios)
  - then another rotation/reflection
- Matrix view: decomposite any m * n matrix M into $$M = U\Sigma V^*$$ (\* for conjugate transpose), where:
  - U and V are orthogonal matrices of size m * m and n * n
  - $$\Sigma$$ is a m * n *rectangle diagonal matrix* with the *singular values* of M on the diagonal ((1,1), (2,2), ..., (min(m, n),min(m, n)))
- There is also a kind of compact SVD where $$\Sigma$$ is a r * r matrix only containing positive singular values (r is the rank of M), and $$U^*U = V^*V = I_{r * r}$$
- Singular values: square roots of the (non-negative) eigenvalues of $$M^*M$$

## Principal Component Analysis

- Dimensionality reduction view:

  - Fitting a k-dimensional ellipsoid to the data
    - i.e. finding out the very best k-dimensional subspace that minimizes MSE
    - Equivalent to finding out k directions that contain most variance of the data
  - Each principal component represents one axis of the ellipsoid

- Factor analysis view:

  - Figuring out *k* factors that explain most of the variance in dataset X
  - Each factor is a configuration representing a linear combination of *p* variables
  - Each obsevation can be quantified by the scores it gets on these factors, i.e. its "similarity" with each configuration

- Formulation: $$X = T_pW_p^\top$$

  - $$X \in \R^{n * p}$$ is the data matrix with *n* observations of *p* *variables* (e.g. *p* entries in a scale)
  - $$W_k \in \R^{p*k}$$ contains (in each column) the *loadings* or coefficients representing *k* *factors/components* in p-dimensional space
    - $$k \le p$$, and in most applications $$k \ll p$$
  - $$T_k \in \R^{n*k}$$ contains the *scores*, i.e., the representation of the data in factor space
    - $$T_k = XW_k$$
    - For approximation, $$X_k = T_kW_k^\top$$

- Computation:

  - By *spectral decomposition* of $$X^\top X$$:

    - the factors *W* are the (unit) eigenvectors of $$X^\top X = W\Lambda W^\top$$ (orthogonal diagonalization of the *covariance matrix* if *X* is centralized)
      - If the range of different variables (columns) in X are significantly different, the decomposition is conducted on the *correlation matrix* rather than covariance matrix, which is similar to z-scoring X before PCA

    - the variances explained are proportional to the eigenvalues

  - By *singular value decomposition* of *X*:

    - $$X^\top X = V\Sigma^\top U^\top U\Sigma V^\top = V \hat{\Sigma^2}V^\top$$, which is exactly $$W\Lambda W^\top$$
    - $$W = V, T = U\Sigma$$
    - We can also consider the k largest singular value and corresponding left and right singular vectors too, in which case $$X_k = U_k\Sigma_kV_k^\top$$ will be the optimal rank-k approximation of X using *Frobenius norm*

## Independent Component Analysis

- General definition:
  - Assumption: the observed data $x = (x_1, x_2, \dots, x_m)^\top\in \R^{m * l}$ is made up by statistically independent hidden signals $s = (s_1, s_2, \dots, s_n)^\top\in \R^{n * l}$, and the value of each signal should be non-Gaussian
  - Task: figure out a linear transformation $W$ as $s = Wx$, so as to maximize some kind of statistical independence measure $F(s_1, s_2, \dots, s_n)$
- Linear form:
  - Assuming that the data is a linear combination of all signals: $x = As$
    - or $x = As + \epsilon$ (Gaussian noise)
    - $x$ is usually centered to zero first
  - $A = (a_1, a_2, \dots, a_n)\in \R^{m * n}$ is called the mixing matrix
    - usually $n = m$
    - however, $n$ can be larger than $m$ (e.g., in cocktail party senario), in which case the problem is overcomplete but still solvable with pseudo inverse
- Calculation:
  - Centering the data
  - Whitening and dimensionality reduction
    - decomposing the data into orthogonal components (and discarding some of them)
    - scaling the scores to unit variance (so the covariance matrix becomes identity matrix)
    - usually with PCA or SVD
  - Iteratively finding out the components (sources):
    - optimization targets:
      - minimizing the mutual information between components
      - maximizing the non-Gaussianity of each component
    - algorithms: Infomax, FastICA, JADE, etc.

- For fMRI:
  - usually the statistical independence is calculated along the sampling dimension (here with length $l$), which often represents time
  - however, in fMRI it is more suitable to calculate along the feature dimension (i.e. you should transpose $x$ before computation), because
    1. in fMRI $m \gg l$
    2. we want to calculate the connectivity between components along the temporal dimension, so they certainly should not be indepedent along this dimension
    3. non-Gaussianity and independence of the components in space means we can find more distinct and interpretable network configurations

