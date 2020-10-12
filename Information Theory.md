# Information Theory

## Intro

- Main question: the amount and transfer speed of information
- Main concept: entroy (the amount of information) and mutual information (bound of the transfer speed)

## Encoding

- Key point: 传输是很多次的并且首尾相连，因此需要前缀不同

- Prefix-free is a sufficient condition for correct decoding, but not necessary.

- Kraft inequality: $$\sum_i 2^{-l_i} <= 1$$, where $$l_i$$ is the coding length of the i-th signal.

- **Minimal coding length**:

  Let $$q_i = 2^{-l_i}$$ and $$p_i = Pr\{occurance\ of\ i\}$$, then average coding length = cross_entropy{p, q}. Therefore the optimal coding is to let $$q_i = p_i$$, and the average coding length is exactly the **Entropy** of the random variable generating this signal.

  Practically, we let $$l_i = \lceil\log p_i\rceil$$, and it can be proved that the coding length is no longer than the optimal coding length + 1bit.

- In some sense, minimal coding length characterized the amount of information one random variable (generating this signal sequence) contains.

## Entropy

- Joint entropy: $$H(X_1, X_2) \le H(X_1) + H(X_2)$$, and the equation suffices when the two variables are independent.
- Conditional entropy: $$H(Y|X) = \sum_{i} P(X = i) H(Y|X = i) = \sum_{i}P(X = i) \sum_{j} -P(Y = j|X = i)\log P(Y = j|X = i)$$, and $$H(Y|X) = H(Y)$$ if the two variables are independent.
- Easily we have $$H(X,Y) = H(X) + H(Y|X)$$.
- **Mutual information:** $$ I(X;Y) = H(X) + H(Y) - H(X,Y) = H(Y) - H(Y|X) = H(X) - H(X|Y)$$.
  - Mutual information is non-negative.
  - It represents the amount of B's information in A or A's information in B. When they are independent, the mutual information is 0. When A = f(B), the mutual information is H(A)?
  - Unfortunately, there is still no satisfactory definition for the mutual information between more than two variables.
- **Relative entropy (KL Divergence):** $$D(p||q) = \sum p_k \log \frac{p_k}{q_k} \ge 0$$. It is the extra bits you used if you encode a random variable with underlying distribution p with the optimal code for distribution q.
- **Data processing inequality: ** $$I(X;Z) \le I(X;Y)$$ if X -> Y -> Z is a Markov process.
- **Entropy rate:** It qualifies the information in a markov process (while entropy only deals with a single random variable). It is defined as $$\lim_{n \to \infin} \frac{1}{n} H(X_1, X_2, \dots , X_n)$$.

## Kolmogorov Complexity

- The K-complexity of a string *S* relative to a turing machine *U* is defined as the minimal length of a program which runs on *U* and outputs *S*.
  - $$K_U(S) \le K_{U'}(S) + C$$, where C is a function of U and U' (length of the program that runs on U' which models U). This means that the complexity is relatively independent of the programming language.
  - $$K_U(S)$$ is incomputable.

