# Geometry

## Riemannian Geometry

- Riemannian geometry is the study of Riemannian manifolds (or Riemannian spaces).

  - A Riemannian manifold $(M, g)$ is a real, smooth manifold $M$ equipped with a positive-definite inner product $g_p$ on its tangent space $T_pM$ at each point $p$.
  - $g$ is called a Riemannian metric. By convention, $g$ is usually also smooth.

- Tangent space projection:

  ![Tangent Space Projection](figure/Tangent Space Projection.png)

  - In order to calculate the distance between two points on the manifold, we need to calculate the speed at each point (according to the inner product there) and integrate them.
  - In a neighbourhood of a reference point $X$, consider a vector in the tangent space $\overrightarrow{XY}$.
    - There will be one and only one geodesic that has the same direction as $\overrightarrow{XY}$ at $X$.
    - There will be one and only one point $Y_0$ on this side of this geodesic such that $|\overrightarrow{XY_0}|_M = |\overrightarrow{XY}|$
    - Therefore, we can map the points in the neighbourhood of $X$ in the tangent space to the points in the neighbourhood of $X$ in $M$.
  - Exponential map:
    - The map above is called an exponential map: $\exp_X(\overrightarrow{XY}) = \gamma_{\overrightarrow{XY}}(1) = Y_0$, i.e. traveling on the geodesic $\gamma_{\overrightarrow{XY}}$ starting from point $X$ with initial speed $\overrightarrow{XY}$ for an unit of time.
    - The inverse of the map is $\overrightarrow{XY} = \log_X(Y_0)$.
    - Note: $Y_0$ is usually just denoted as $Y$, but be careful that it sometimes represents the point on the manifold and sometimes that on the tangent space.
      - The reason to do that is to re-interpret some basic operations on the Riemannian manifold to simplify the calculation.
      - ![Reinterpretation](figure/Exp Log Riemmanian.png)
    - Why this map is called "exponential":
      - for the manifold of all orthogonal matrix $M$, the exponential map at the point of identity matrix $I$ from the tangent space $M_I$ (which is exactly the manifold of all asymmetric matrix) is $\exp_I(A) = e^A = \sum_{m = 0}^\infin \frac{A^m}{m!}, \forall A \in M_I$
      - it is also true for our analysis below

- Application in covariance-matrix-based predictions:

  - The set of all possible covariance matrix isn't a vector space, since $A, B \in U $ does not imply $A - B \in U$. Instead, they form a Riemannian manifold $sym_n^+$ (i.e., all n\*n symmetric positive definite matrix).

  - Probablistic model based on covatiance matrices:

    - Each subject's covariance matrix can be modeled as the sum of the group mean and a deviation: $\Sigma^s = \Sigma^* + d\Sigma^s$, where $d\Sigma^s$ has non-Euclidean geometry.

    - Therefore, the probablistic model will be more accurate if they take the Riemannian manifold structure into consideration.

    - A generalized Gaussian model for covariance matrices:
      $$
      p(\Sigma) = k(\sigma)\exp(-\frac{1}{2\sigma^2}||\overrightarrow{\Sigma^*\Sigma}||^2_{\Sigma^*})
      $$
      where $\Sigma^*$ is the norm defined at point $\Sigma^*$, $\sigma$ is an isotropic (各向同性的) variance over the manifold and $k$ is a normalization factor.

  - One way to handle this structure is to project the covariance matrix into the tangent space:

    - The manifold $sym_n^+$ itself is not a vector space, thus having non-Euclidean inner product. 
    - However, the tangent space is a vector space, so we can use normal 2-form inner product on the tangent space to indirectly calculate the distance on the manifold.
    - It has been proved that $\phi_A: B \to \log(A^{-\frac{1}{2}}BA^{-\frac{1}{2}})$ can map the point (matrix) $B$ to a vector (another matrix) $\overrightarrow{AB}$ in the tangent space, such that $||\overrightarrow{AB}||^2_A = ||\log(A^{-\frac{1}{2}}BA^{-\frac{1}{2}})||^2_2$
      - Similarly, $B = \phi^{-1}_A(\overrightarrow{AB}) = A^{\frac{1}{2}}\exp(\overrightarrow{AB})A^{\frac{1}{2}}$

  - Estimation of model parameters:

    - We can rewrite the distribution using $d\Sigma^s = \log_{\Sigma^*}(\Sigma^s) \in T_{\Sigma^*}M$ rather than $\Sigma^s$:
      $$
      p(d\Sigma^s) = k’(\sigma)\exp(-\frac{1}{2\sigma^2}||d\Sigma^s||^2_2)
      $$
      which is a normal distribution defined on a flat vector space $T_{\Sigma^*}M$ with diagonal covariance $\sigma$.

    - When the distribution is narrow, i.e. $||d\Sigma^s||^2_2 \ll 1$, we have:
      $$
      \Sigma^s = \phi^{-1}_{\Sigma^*}(d\Sigma^s) = {\Sigma^*}^{\frac{1}{2}}\exp(d\Sigma^s){\Sigma^*}^{\frac{1}{2}} \approx {\Sigma^*}^{\frac{1}{2}}(I_n +d\Sigma^s){\Sigma^*}^{\frac{1}{2}}
      $$
      (since the exponential map has the same form as matrix exponential)

    - The maximum likelihood estimation of the intrinsic mean $\Sigma^*$ is the Frechet mean $\hat{\Sigma^*} = \arg\min_{\Sigma^*} \sum_s ||\overrightarrow{\Sigma^*\Sigma^s}||^2_{\Sigma^*}$

      - see the algorithm 3 in [(Riemannian geometry for the statistical analysis of diffusion tensor data)](http://www.sci.utah.edu/~fletcher/FletcherTensorStatsSP.pdf)
      - this algorithm uses the exponential and logarithmic maps in each step of the iteration for each of the individual matrix, which can be computed with the first two algorithm in the same paper (need to use eigenvalue decomposition)
      - the complexity should be similar to $O(n^3\times S\times\text{iter})$

    - Then we can estimate the $d\Sigma^s$ by $\hat{d\Sigma^s} = \hat{\Sigma^*}^{-\frac{1}{2}}\Sigma^s\hat{\Sigma^*}^{-\frac{1}{2}} - I_n$, which give us the tangent space projection of each subject $\hat{\Sigma^*} + \hat{d\Sigma^s}$

    - Besides, the maximum likelihood estimation of $\sigma^2$ is $\hat{\sigma^2} = \frac{1}{S}\sum_s||\text{Vec}(d\Sigma^s)||^2_2$, where $S$ is the number of subjects and $\text{Vec}$ is the vectorize operator

## Topological Data Analysis

- Concept:
  - using algebraic topology to figure out the topological and geometrical features in point clouds
  - algebraic topology: to find algebraic invariants (using abstract algebra) that classify topological spaces up to homeomorphism

- Pipeline:
  - Starting from a point cloud with associated pairwise distance matrix
  - Construct topological structures on the data:
    - usually simplical complexes or nested family of simplical complexes (called filtration)
    - can be understood as high-dimensional generalizations of neighbouring graph
  - Extract information from the structures, e.g.:
    - a full triangulation of the underlying shape of data
    - a coarse summary, e.g. persistent homology
  - Visualize or analyze the data using the extracted information

- Construct simplical complexes from data:
  - The constructed complex is heavily influenced by the scale of analysis, which is the basis of persistent homology

  - Given a certain scale $a$, we can define a complex on a data cloud $X$ in two ways:
    - Vietoris-Rips complex is a set of simplical complexes $\{x_0, \dots, x_k\}$ where $d_X(x_i, x_j) \le a$.
      - Here the distance between two sets are defined in a min-max fashion.
      - Note that $i$ can be the same as $j$, restricting the size of the simplical complex.
    - Cech complex is a set of simplical complexes where "inflating" them (creating a close ball centered around each point) by $a$ leads to a common intersection.
    - $Rips_\alpha(X)\sube Cech_\alpha(X) \sube Rips_{2\alpha}(X)$

    <img src="https://www.frontiersin.org/files/Articles/667963/frai-04-667963-HTML-r3/image_m/frai-04-667963-g002.jpg" alt="img" style="zoom:33%;" />

    ​	($Rips_\alpha(X)$ on the left and $Cech_\alpha(X)$ on the right)

- Homeomorphism (同胚), homotopy (同伦), and homology (同调):

  - Homeomorphism:

    - Two spaces $X$ and $Y$ are said to be homeomorphic, if there exist two continuous bijective maps $f: X\to Y,g: Y \to X$ such that $f\circ g$ and $g\circ f$ are the indentity map of $Y$ and $X$ respectively.

  - Homotopy:

    - Two maps $f_0, f_1: X\to Y$ are said to be homotopic if there exists a continuous map $H:X\times [0, 1] \to Y$ such that for any $x\in X$, $H(x, 0) = f_0(x), H(x, 1) = f_1(x)$

    - You can imagine the last variable in $H$ as time, e.g., here is a homology between the surface of a coffee mug and the surface of a doughnut:

      - $X$ is a torus, $Y$ is $R^3$
      - $f$ is a continuous function that maps the torus to the surface of the coffee mug
      - $g$ is a continuous function that maps the torus to the surface of the doughnut
      - $H$ is the "deformation"  process

      <img src="https://upload.wikimedia.org/wikipedia/commons/2/26/Mug_and_Torus_morph.gif" alt="img" style="zoom: 50%;" />

  - Homotopy equivalent:
    - Two spaces $X$ and $Y$ are said to be homotopy equivalent if there exists two continuous maps $f:X\to Y, g:Y\to X$, such that $f\circ g$ and $g\circ f$ are homotopic to the identity map in $Y$ and $X$ respectively. And the two maps $f,g$ are said to be homotopy equivalent.
    - Intuitively, two spaces are homotopy equivalent if one can be reformed into another by expanding, shrinking and bending
    - Spaces that are homotopy equivalent to a point is called contractable.
    - Homeomorphism is a special case of homotopy equivalence. For example, a solid disk (as well as $R^n$) is homotopy equivalent to a point, but not isomorphic to it (no bijection).
  - Homology:
    - Informally, the homology of a topological space $X$ is a set of topological invariants of $X$ represented by its homology groups $\{H_0(X), H_1(X), \dots\}$ where $H_k(X)$ describes the number of k-dimensional holes in $X$

- Nerve theorem:
  - Given a cover $\mathcal  U = (U_i)_{i\in I}$ of $M$, the nerve of $\mathcal U$ is a simplical complex $C(\mathcal U)$ with:
    - $\mathcal U_i$s as the vertices
    - certain sets of $\mathcal U_i$s as facets, if and only if all $\mathcal U_i$s in the set have shared intersections (i.e., such set might contain two or three or more $\mathcal U_i$s)
  - Nerve theorem: Let $\mathcal U = (U_i)_{i\in I}$ be a cover of a topological space $X$ by open sets such that the intersection of any subcollection of the $U_i$s is either empty or contractible. Then, $X$ and the nerve $C(U)$ are homotopy equivalent.
  - In particular, if $X$ is a set of points in $R^d$, then the Cech complex $Cech_\alpha(X)$ is homotopy equivalent to the union of balls $\cup_{x\in X}B(x, \alpha)$

- the Mapper algorithm
  - Refined pull-back cover:
    - Given a continuous real-valued function $f: X\to R^d (d\le 1)$ and a cover of $X$ $\mathcal U = (U_i)_{i\in I}$, the pull-back cover of $X$ induced by $(f, \mathcal U)$ is the collection of $f^{-1}(U_i)$
    - The refined pull-back cover is the collection of connected components (defined by clustering algorithms) of $f^{-1}(U_i)$
  - The Mapper algorithm summarizes the data by the nerve of the refined pull-back cover. With appropriate choice of $f$ and $\mathcal U$, the nerve is usually as simple as a graph.
  - $f$ is sometimes called the filter or lens function. There are many choices of it, e.g. density estimates, PCA coordinates, etc.
  - $\mathcal U$ is usually defined as a set of regularly placed intervals with equal length.
    - The length $r$ is called the resolution of the cover.
    - The percentage of overlap $g$ is called the gain of the cover. If $g < 50%$, the output nerve is certainly a graph.
    - The mapper algorithm is very senstive towards the choice of $\mathcal U$, so exploration of different choices is necessary.