# Machine Learning

## Reinforcement Learning

- Concept:

  - Reinforcement learning (RL) is one of the three major types of learning, along with supervised and unsupervised learning
  - Typical RL is modeled as a Markov Decision Process:
    - at each timepoint $t$, an agent performs an action $a_t \in A$, receiving an associated reward $r_t(a_t)$ and transiting from state $s_t \in S$ to $s_{t + 1} \in S$
    - $P_a(s, s') = \text{Pr}\{s_{t + 1} = s'|s_t = s, a_t = a\}$ is the probability of transiting to state $s'$ from $s$ by action $a$, and $R_a(s, s')$ is the associated reward (or its expectation if it's random)
    - The agent acts according to a policy $\pi(a, s) = \text{Pr}\{a_t = a | s_t = s\}$, the goal is to find an optimal (or near optimal) policy that maximizes the expected accumalted rewards, usually with infinite time horizon
    - Since $s_{t+1}$ only depends on $s_t$ and $a_t$, while $a_t$ only depends on $s_t$, ultimately $s_{t+1}$ only depends on $s_t$, so the process is a Markov Process
  - Observability: if the agent only has access to part of the states or if the observed states are corrupted by noise, the problem becomes a partially observable Markov decision process
  - Regret: the difference between the accumulated rewards of current policy and the optimal policy

- Value function learning:

  - Estimate the value of each state and the optimal policy

  - Only useful when you have full knowledge of the reward and state transition function

  - State value function: the expected discounted accumulative rewards at state $s$ with policy $\pi$
    $$
    V_\pi(s) = \text{E}\left [ \sum_{t = 0}^\infin \gamma^t r_t | s_0 = s \right ]
    $$
    where $\gamma\in [0, 1)$ is called the discount-rate.

  - Using Bellman equation:
    $$
    V_\pi(s) = \text{E}\{r_0 + \gamma V_\pi(s_1)|s_0 = s\}
    $$

  - Starting from a random guess of $V(s)$ and $\pi(s)$:
    $$
    V(s) \leftarrow \sum_{s'}P_{\pi(s)(s, s')}(R_{\pi(s)}(s, s') + \gamma V(s'))\\
    \pi(s) \leftarrow \arg\max_\alpha \left \{\sum_{s'}P_{\alpha}(s, s')(R_{\alpha}(s, s') + \gamma V(s'))\right \}
    $$
    And there are many versions of algorithms depending on the order to do step 1 and step 2.

- Temporal difference learning (TD)

  - TD is a way to learn the value of each state under a *given* policy, without knowledge of the rewards and state transition functions

  - TD-learning: starting from a random table of $V(s)$ and update it online by
    $$
    V(s) \leftarrow V(s) + \alpha (\underset{\text{TD error}}{\overbrace{r + \gamma V(s')}^{\text{TD target}} - V(s)})
    $$
    where $\alpha > 0 $ is a learning rate, $s'$ is the next step chosen by the policy, and $r$ is the associated reward.

  - The dopamine neurons in ventral tagmental area fires in resemblance of the TD error (the prediction error theory)

- Q learning:

  - Q learning is a way to learn the quality of state-action pairs
    - thus a way to find out the optimal policy without knowledge of the reward and state transition functions

  - Action value function: the expected discounted accumulative rewards for choosing action $a$ at state $s$ and continue with the optimal policy $\pi^*$
    $$
    \begin{aligned}
      Q(s, a) & = \text{E}_{s_1}\left \{R(s, s_1) + \sum_{t = 1}^\infin \gamma^tr^*_t \right \}\\
      		  & = \text{E}_{s_1}\{r_0 + \gamma V^*(s_1) \}\\
      		  & = \text{E}_{s_1}\{r_0 + \gamma \max_{a'}Q(s_1, a') \} \\
    \end{aligned}
    $$
  
  - Therefore, the Q function can be learned iteratively:
    $$
    Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left (r_t + \gamma \max_a Q(s_{t + 1}, a) - Q(s_t, a_t)\right )
    $$
  
  - However, it's hard to simply learn the function by filling a table of all possible $(s, a)$, where a lot of combinations will never be visited. Therefore, $Q$ is often acquired using some function estimation techniques, e.g., an artificial neural network trained by the squared TD error (deep Q network, DQN)


## Transformers

- Concepts:
  - *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)*, Ashish Vaswani et. al., 2017
  - RNN (including LSTM/GRU, same below) can hardly remeber everything it needs when the dependency is too long-range. Besides, it can hardly be parallelized.
  - What if we feed in all data (a sentence or even an article) at one time?
    - No recurrence, hightly paralleled
    - Dependency is extracted by *self-attention*
    - Position is encoded seperately

- Self-attention:

  - Attention:

    - [(Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio, 2014)](https://arxiv.org/abs/1409.0473) & [(Thang Luong et. al., 2015)](https://arxiv.org/abs/1508.04025v5)

    - The idea is to "align"  the tokens in inputs and model outputs that contains relevant information but may have different relative position in the sequence.

    - Generally speaking, for each input token (in RNN, it's the same as a time step), an attention layer receives these inputs:

      - a vector $$c$$ representing the context at current token
      - a collection of the outputs from the previous layer for all nearby tokens $$\{y_t | t\in U\}$$, where U may contain a short sequence before (forward RNN) or/and after (backward RNN) the current token

    - Essentially, attention layer performs a weighted summation over the past or/and future states $$\{y_t\}$$, with weights of each states selected according to their relevance to the current context $$c$$

      <img src="https://miro.medium.com/max/817/1*ol7jlD1cGbUHawotKphD-w.png" alt="Attention" style="zoom:50%;" />

      - The summation (the *alignment function*) is usually implemented with softmax
      - The relevance (the *score function*) can be measured in many ways, usually addition or multiplication after kernel transformation

  - Self-attention:

    - QKV model of attention mechanism: a view of database query
      - spliting the input $$y$$ into two parts: key $$k$$ and value $$v$$
      - rename context $$c$$ to query $$q$$
      - relevance is calculated between $$k$$ and $$q$$
      - alignment is applied onto $$v$$
    - $$q, k, v$$ are all linear transformations of the input $$y$$, usually of the same dimension
    - Score is implemented as dot product.
    - "Receptive field" (range of relevant tokens) is the whole input (e.g. a sentence, an article)

  - Multi-head attention:

    - Use different projections to generate multiple sets of $$q, k, v$$, attending to different aspects of the context
    - Then concatenate the outputs from all heads and project it back to the input space $$\R^d$$
    - Usually when $$y\in \R^d, q, k, v \in \R^{d/a}$$, where $$d$$ is the model width (input embedding dimension & hidden dimension in each block) and $$a$$ is the number of heads

- Encoder-decoder structure:

  - It is a kind of seq2seq model (transform an input sequence to another sequence, e.g. translation).

  - It inherits some traits of the autoencoder network (3-layer perceptron that aims to generate the same output as the input, usually using a low-dimensional hidden layer which extracts the low dimensional embedding of the data)

  - Basically, it encodes the sequence with an encoder and decode its output with a decoder.

  - The structure of the original Transformer:

    <img src="https://pytorch.org/tutorials/_images/transformer_architecture.jpg" alt="../_images/transformer_architecture.jpg" style="zoom:33%;" />

    - Encoder and decoder are both stacks of encoding/decoding units.
    - Each unit contains an attention layer and a fully connected layer.
    - Decoding unit further contains an attention layer for the output of the very top encoding unit.
    - Each layer's output is added up with the residual of the previous layer's and normalized, similar to the idea of ResNet (layers only need to learn the "refinement" of the representation).

  - Positional encoding:

    - added up together with the input
    - can be a trainable linear embedding layer (e.g. in BERT), just like word embedding
    - can be as simple as a sine function (e.g. in original paper)

- Bidirectional Encoder Representations from Transformers (BERT):

  - [(Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, 2018)](https://arxiv.org/abs/1810.04805), developed by Google
  - Outperforms human in reading comprehension dataset SQuAD1.1
  - Structure: almost the same as the original one
    - 12 layers (6 in each of encoders/decoders)
    - 768 dimensional hidden state in each layer, 3072 neurons in feedforward layer
    - 12 attention heads
    - ~110 million parameters (similar to GPT-1)
    - pre-trained on datasets with 3G words
  - Self-supervised learning:
    - Pretraining with self-generated labels (e.g., *Cloze* task, see below) from the data without manual labeling
    - Supervised learning on specific tasks
  - Pre-training:
    - Mask Language Modeling (MLM): essentially the *Cloze* task (完形填空), randomly masking some words in the input and predicting them
    - Next Sentence Prediction (NSP): given a pair of sentence, predicting whether they are consecutive
    - Datasets: *BooksCorpus* (800M words) and English Wikipedia (2500M words)
  - Fine-tuning:
    - State-of-the-art (SOTA) on 11 tasks
    - Usually only few fine-tuning (a few GPU hours) and add one linear output layer
  - Application: nearly all English queries in Google search are processed by BERT

- More recent transformer models from Google:

  - Google T5
    - [(Colin Raffel et. al., 2019)](https://arxiv.org/abs/1910.10683)
    - SOTA on both NLP and Natural Language Generation (NLG); while BERT is only good at NLP
    
  - AlphaFold2:
    - [(John Jumper et. al., 2021)](https://www.nature.com/articles/s41586-021-03819-2)
    - use 2D embeddings to capture the pairwise relationship between units
    
  - Switch transformer:
    - [(William Fedus et. al., 2021)](https://arxiv.org/abs/2101.03961)
    
    - 1.6 trillion parameters
    
    - Sparse activation of the feedforward layer using Mix-of-Expert (MoE) technique
      - Generally speaking, devide the feedforward layer into multiple (here 128) parts (*experts*) and only activate one of them each time using a router, thus reducing computational cost
      - Load is balanced among the experts by modifying loss function
      
    - Further development: [Google Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/)
    
    - Relationship weith Andre Bastos & Nancy Kopell's [predictive routing model](https://www.pnas.org/content/117/49/31459)?
    
      <img src="https://www.pnas.org/content/pnas/117/49/31459/F6.large.jpg" alt="Predictive Routing" style="zoom:25%;" />
    
      - Feedforward gamma oscillation in superficial layer encodes sensory information for current stimulus.
      - Feedback alpha/beta oscillation in deep layer encodes predicted stimulus.
      - Prediction error occurs from the feedforward processing of unprepared (not inhibited) inputs (right panel), not from a specialized circuit that computes the difference between predicted and observed stimuli.
      - Columns (computational modules) for different stimuli may be understood as different "experts"?
        - Feedback signal inhibits the processing of predicted stimuli and enhances that of unpredicted stimuli through selective inhibition of certain modules.
        - In transformer structure, this feedback prediction may be computed directly by multiplying recent states with the router kernel
      - Will a more complex router generate better prediction and help the network perform even better?

- Generative Pre-trained Transformer (GPT):

  - History:

    - Versions:
      - GPT-1: [(Alec Radford et. al., 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
      - GPT-2: [(Alec Radford et. al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
      - GPT-3: [(Tom B. Brown et. al., 2020)](https://arxiv.org/pdf/2005.14165.pdf)
    - Ownership:
      - Developed by OpenAI
      - GPT-1 is open-sourced
      - Only a tiny version of GPT-2 is released
      - GPT-3 is exclusively owned by Microsoft, but API is provided to the public
    - OpenAI:
      - Founded by Elon Musk (who resigned later) and others in 2015 as a non-profit laboratory
      - Turned for-profit later
      - Co-founder and chief scientist: Ilya Sutskever, co-inventor of AlexNet
      - Co-founder and CEO: Sam Altman, Saint-Louis-raised entrepreneur and angel investor
      - Now receiving billions of investment from Microsoft

  - GPT-1:

    - Structure:

      <img src="https://engineering.linecorp.com/wp-content/uploads/2019/01/Fine-tuning.png" alt="img" style="zoom:50%;" />

      - 12 layers, only the decoder part is used
      - 768 dimensional hidden state, 3072 dimensional feedforward layer
      - 12 attentional heads

    - Pre-training:

      - Trained left-to-right, maximizing the output probability of the next token given previous tokens
      - Trained on BooksCorpus

    - Finetuning:

      - see figure above, usually can be done withtin 3 training epochs
      - objective function is the linear combination of task objective and language model (next token prediction) objective

    - Performance: good zero-shot performance on various tasks

  - GPT-2:

    - Task conditioning and zero-shot transfer learning:
      - Use a single structure to do all tasks
      - Learning objective is no longer P(output|input) but P(output|input, task)
      - Task is input by verbal instructions along with other materials, and the model is supposed to figure it out from the text and generate corresponding output
      - No finetuning or extra output layer is needed
    - Structure:
      - 48 layers, 1600 dimensional embedding
      - larger dictionary with 50,257 tokens
      - larger context window of 1024 tokens
      - layer normalization is moved to the input of each block and an extra layer normalization on top of the final self-attention block
      - 1.5 billion parameters
    - Pre-training:
      - 40GB data from web crawl
      - Wikipedia is excluded since it is the source of many testing sets
    - Performance: beating SOTA in 7 of 8 tasks without finetuning

  - GPT-3:

    - In context-learning:

      ![GPT-3 Transfer Learning](figure/Machine Learning 1.png)

      (from [GPT-3 paper](https://arxiv.org/pdf/2005.14165.pdf))

      - finetune the model with more examples provided without updating parameters
      - the model is supposed to match the given patterns to its knowledge and understand the task better given more examples
      - indeed, the model performs better in many tasks in few-shot rather than zero-shot or one-shot settings

    - Structure:

      - 96 layers of 12,288 dimension, each with 96 attentional heads of 128 dimension
      - larger context window of 2048 tokens
      - attention patterns are modified to be alternating dense and sparse
      - 175 billion parameters

    - Pre-training:

      - trained on a dataset of 570GB (~400 billion byte-pair-encoding tokens) filtered from 45TB plaintext
        - 82% from web crawl data (60% *Common Crawl* and 22% *WebText2*)
        - 16% from *Bookscorpus*
        - only 3% from Wikipedia
      - training is so expensive that researchers couldn't afford to re-train the model after finding out a bug that introduces some testing data sources into training set (influences seemed to be small though)

    - Performance:

      - beating SOTA without finetuning in many tasks
      - passing the Turing test in some generative tasks
      - performing some tasks that is never trained on:
        - arithmatic summation
        - writing codes like SQL query and Javascript given natural language description
        - unscrambling words in a sentence
        - ...

    - Some interesting applications:

      - *the guardian* asked GPT-3 to write an [essay](https://www.theguardian.com/commentisfree/2020/sep/08/robot-wrote-this-article-gpt-3) to prove that it is not harmful to human, given an introduction paragraph; the outputs are quite good and comparable to human
      - currently there are ~300 [applications](https://openai.com/blog/gpt-3-apps/) using GPT-3 API

  - Further development:

    - DALL-E: 
      - [(Aditya Ramesh et. al., 2021)](https://arxiv.org/abs/2102.12092)
      - zero-shot (natural language) text-to-image generation
      - using 12 billion parameter version of GPT-3
    - CLIP:
      - [(Alec Radford et. al., 2021)](https://arxiv.org/abs/2103.00020)
      - zero shot image classification based on natural language prompts (e.g. using different categories)

- *Wu Dao* (悟道):

  - a large pre-trained multimodal transformer [model](https://wudaoai.cn/home) developed by *Beijing Academy of Artificial Intelligence* (北京智源) 
  - *Wu Dao* 2.0 was announced in May 2021 and contains 1.75 trillion parameters
  - also using MoE
  - trained on 4.9TB of images and text, including 1.2TB of Chinese text
  - incorporates multiple models including language models, image captioning models, and protein folding prediction models
  - outperform SOTA in 9 language modeling, image classification and image captioning tasks

## Generative Models

- In statistical classification and modeling, considering observable variable $X$ and target variable $Y$, two types of model can be defined:
  - Generative models, the models for the joint distribution $p(X, Y)$, or to a less degree (as is enough for many scenarios), the models for the conditional distribution $p(X|Y)$
  - Discriminative models, the models for the conditional distribution $p(Y|X)$
  
- "Generative":
  - Generative models can be used to simulate new data using the conditional distribution $p(X|Y)$
  - Therefore, the word "generative model" are also used to refer to the models that can generate certain types of data in a way that has no clear relationship with the input distribution (e.g., using random seeds)
  - In these models, the output is usually judged by the similarity against the data's distribution, so you can think of $Y$ as an intermedia (the "seed" above) between data $X$ and the output $X'$
  
- Variational autoencoder (VAE):

  <img src = "https://upload.wikimedia.org/wikipedia/commons/1/11/Reparameterized_Variational_Autoencoder.png" style = "zoom:40%"/>

  - Scenario: similar to EM algorithm

    - Assuming that data $x$ is generated by some random process involving some latent variables $z$ with prior $p_{\theta^*}(z)$ and conditional distribution $p_{\theta^*}(x|z)$
    - $p_{\theta^*}(z)$ and $p_{\theta^*}(x|z)$ come from families $p_\theta(z)$ and $p_\theta(x|z)$
    - Intractability:
      - The marginal $p_\theta(x) = \int_z p_\theta(x|z)p_\theta(z)dz$ is intractable
      - as well as the posterior $p_\theta(z|x)$ (so that EM can not be used)
      - as well as any other required integrals for mean-field variational bayes algorithm

  - The goal:

    - Efficiently infer $\theta^*$ from data $x$
    - Efficiently approximate the posterior $p_{\theta^*}(z|x)$
    - Efficiently approximate marginal inference $p_{\theta^*}(x)$

  - The model:

    - A probablistic encoder learns a recognition model $q_\phi(z|x)$ that approximates the true posterior $p_\theta(z|x)$
      - i.e., the output of the encoder given input $x_i$ is the parameter $\phi$ (e.g., mean, variance) for the distribution $q_\phi(z|x_i)$
    - A probablistic decoder learns the generative model $p_\theta(x|z)$
      - i.e., the output of the decoder given hidden code $z_j$ is the parameter $\theta$ for the distribution $p_\theta(x|z)$

  - Training:

    - We want to approximate $p_\theta(z|x)$ by $q_\phi(z|x)$, as well as to infer the optimal $\theta$ by $p_\theta(x)$, so we use the Evidence Lower Bound (ELBO) loss function:
      $$
      \begin{aligned}
      L_{\theta, \phi} & = -\underset{\text{log evidence}}{\underline{\log(p_\theta(x))}} + \underset{\text{Non-negative KL divergence}}{\underline{D_{KL}\left ( q_\phi(z|x) || p_\theta(z|x)\right )}} \\
      	& = -\underset{\text{Decoder}}{\underline{E_{z \sim q_\phi(z|x)}[\log(p_\theta(x|z))]}} + \underset{\text{Encoder}}{\underline{D_{KL}(q_\phi(z|x) || p_\theta(z))}} \\
      	& = E_{z \sim q_\phi(z|x)}[\log (q_\phi(z|x)) - \log(p_\theta(x, z))]
      \end{aligned}
      $$

    - Note that:

      - The above deduction is general for all variation bayesian inference problems
      - $\log(p_\theta(x)) \ge -L_{\theta, \phi}$, so the negative of ELBO provides a lower bound to the log evidence
      - $L_{\theta, \phi}$ is also called *variational free energy* analogous to the thermodynamical free energy, since it is the sum of:
        - the entropy of the distribution $q$
        - a negative energy term $-E_{q}[\log p(x, z)]$

    - However, we can not directly perform stochastic gradient descent on the expetation term

      - The gradient operator for $\theta$ commutes with the expectation operator, so we can perform the gradient descent at single sample level using $\frac{1}{L}\sum_{l = 1}^L\grad_\theta\log(p_\theta(x_i, z_{i, l}))$, where $z_{i, l}$ are the l-th sample drawed from $q_\phi(z|x_i)$ using Monte Carlo methods (usually L = 1)
      - However, the gradient operator for $\phi$ does not commute with the expectation, which itself involves $\phi$

  - Tricks:

    - The conditional distribution $q_\phi(z|x)$ is usually set as a very simple distribution
      - e.g., a multivariate normal distribution with diagonal variance $z\sim \mathcal{N}(\mu, \sigma^2)$
      - The complexity of the dependency between $x$ and $z$ can be compensated by the function approximation of $p_\theta(x|z)$, which is done by neural networks
    - Random sampling $z\sim q_\phi(z|x)$ can be replaced by a deterministic function with a noise input
      - In case of Gaussian distribution, it can be $z = \mu + \sigma \odot \epsilon$, where $\odot$ is element-wise product and $\epsilon \sim \mathcal{N}(\bold{0}, \bold{1})$ is a noise input
      - In this case, the gradiant operator can commute with the expectation (which depends on $\epsilon$ rather than $\phi$ now)
      - In this case, the encoder approximates $\mu$ and $\sigma$ and can be trained with stochastic gradient descent

- Generative Adversial Network (GAN):

  - A generator and a discriminator contest each other in the form of zero-sum game
  - The discriminator is just a classifier (e.g., a convolutional neural network for images), while the generator maps a latent variable (usually a multivariate normal distribution) to the output (e.g., a deconvolutional neural network)
  - The two networks are trained iteratively:
    - The generator's outputs are sent to the discriminator along with some real images
    - The generator's goal is to improve the error rate of the discriminator, while the discriminator's goal is to reduce it
    - The discriminator is usually trained multiple times in an iteration and with a larger learning rate, so as to provide a better judgement of the performance of the generator at the begining

  - At the end, the discriminator's accuracy drops to 50% and the training stops

## Linear Methods

### Linear Discrimination Analysis

- Generally speaking, LDA is a way to classify the data generated from a Gaussian model:

  - Assuming $K$ categories each with prior probability 
  - The likelihood function for each categories is a multivariate normal distribution $f_k(x) = \frac{1}{(2\pi)^{p/2}|\Sigma_k|^{1/2}}\exp[-\frac{1}{2}(x - \mu_k)^\top\Sigma_k^{-1}(x - \mu_k)]$, where $x\in \R^p, k = 1, 2, \dots, K$

- The classification can be based on:
  $$
  \frac{\Pr(G = k | X = x)}{\Pr(G = l | X = x)} = \frac{f_k(x)\pi_k}{f_l(x)\pi_l}
  $$

- If $\Sigma_1 = \Sigma_2 = \dots = \Sigma_K = \Sigma$, we have:
  $$
  \begin{aligned}
  \log\frac{\Pr(G = k | X = x)}{\Pr(G = l | X = x)} & = \log\frac{\pi_k}{\pi_l} + \log\frac{f_k(x)}{f_l(x)} \\
    & = \log\frac{\pi_k}{\pi_l} - \frac{1}{2}(\mu_k + \mu_l)\Sigma^{-1}(\mu_k - \mu_l) + x\Sigma^{-1}(\mu_k - \mu_l) \\
    & = \log\frac{\pi_k}{\pi_l} + x\Sigma^{-1}(\mu_k - \mu_l) - \frac{1}{2}(\mu_k \Sigma^{-1}\mu_k - \mu_l \Sigma^{-1}\mu_l)
  \end{aligned}
  $$
  Therefore, we can define a *linear discriminant function*:
  $$
  \delta_k(x) = \log\pi_k + x\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k\Sigma^{-1}\mu_k
  $$
  And the decision rule is equivalent to finding $\arg\max_k \delta_k(x)$.

- If the covariance matrix is not the same for each category, we will need to define a *quadratic discriminant function* (QDA):
  $$
  \delta_k(x) = -\frac{1}{2}\log|\Sigma_k| + \log\pi_k - \frac{1}{2}(x-\mu_k)^\top\Sigma^{-1}(x-\mu_k)
  $$

### Regularization

- Regularization is useful when the samples are relatively limited compared the number of parameters:
  
  - Preventing overfitting
  - Stablizing the training result
  
- Add a regularization term to the loss function to restrain the magnitude of the coefficients:
  $$
  L(w, x, y) = L_0(w, x, y) + \lambda R(w)
  $$
  where $L_0$ is the original loss function, $w$ is the weight vector, $R$ is a regularization term and $\lambda$ is a factor controling the extent of regularization, called shrinkage factor.

- Note: the intercept should not be subjected to penalization

- Ridge regression: similar to standard linear regression but with L2 regularization of the slopes
  $$
  L = \sum_{i = 1}^n (y_i - \beta_0 - \sum_{j = 1}^p \beta_jx_{ij})^2 + \lambda\sum_{j = 1}^p \beta_j^2
  $$

- Lasso regression: similar to ridge regression but using L1 regularization instead of L2

- Penalization can also be applied to classification problems:
  $$
  L(\beta, \theta) = \frac{1}{n}||Y\theta - X\beta||^2 + \lambda R(\beta)\\
  s.t.\ \theta^\top Y^\top Y\theta = 1
  $$
  where $Y \in \R^{n * K}, Y_{ik} = 1$ if $y_i = k$, and $K$ is the number of categories.

  Minimizing $L$ will generate a weight vector $\beta$ that is proportional to the one in linear classifiers.




## 规则学习与决策树

### 顺序覆盖法

- 一种贪心方法
- 先学一个规则保证能覆盖部分正样本但不覆盖任何负样本，然后去掉已经覆盖的正样本再继续学

### Learn One Rule

每次学习一条规则：

- Top down：逐渐加约束条件（按某种规则生成，比如按正确率深搜，尽量不覆盖负样本且尽可能覆盖正样本），直到再增加条件无法覆盖更多正样本（比如已经正确率100%了）
- Bottom up: 先取一个实例作为条件，然后逐渐删除条件直至覆盖了负样本

学完后删掉已覆盖的样本。

### Beam Search

- 同时维护k个策略并生成其后续策略，然后在所有策略中挑k个最好的
- 部分解决贪心法局部最优问题

### Decision Tree

- 学习最小决策树是NP完全的
- Entropy的引理：$$H(S) = C \sum p_i \log p_i$$是唯一满足以下三条性质的函数：对于pi连续、对等概分布的n单调递增、将S分成连续做两个试验后原来的H等于两个试验的H的加权和。
- Jensen不等式
- 根据infomation gain选择分裂方式

### 剪枝方法

- 前剪枝：可以在包含的实例太少时剪枝，或者再分下去得不到收益或实例分布与属性取值独立时剪枝
- 后剪枝：通过子树替换（直接并入所有子节点）和子树提升（交换父节点和占比最大的子节点）实现，其依据可以是误差估计或最小描述长度准则
- 最小描述长度准则：$$MDL = L(Data|Model) + L(Model)$$，L代表用于编码的最小比特数。MDL越小，说明数据被压缩得越好，也就是越能抓住数据的本质。
- 误差估计的方法：
  - 验证集误差（MDL）
  - 训练集误差置信区间（C4.5方法）：采用二项分布，则总体错误率的均值和方差的形式已知，可以根据样本均值估计置信区间，然后使用区间上界作为标准，如果父节点的误差上界小于子节点的加权和则合并子节点
- 防止子树过多：引入Split Information，使用Gain Ratio而非简单的Entropy Gain作为裂变标准
  - 但也有问题，如果分裂数很少而且极其不均匀，则分母会特别小



## 循环神经网络

### 计算图模型

- 动力系统：$$s_{t + 1} = f(s_t, \theta)$$
- 输入驱动的动力系统：$$s_{t + 1} = f(s_t, x_{t + 1}, \theta)$$

### BPTT

- 使用交叉熵loss



## 半监督学习

同时使用带标签和不带标签的数据训练

- Self-learning：假设预测时置信度较高的预测是对的，把预测标签加入训练集
- Co-training：把数据按特征维度分为两个集合，训练两个分类器，然后把各自预测置信度最高的样本（的对应特征）加入对方的训练集训练
- MultiView：训练多个不同类型的分类器，对未标注数据投票然后加入训练集



## EM算法

- 背景：X是Z的函数（确定性，已知），二者都由参数θ控制，通过X的观测值推测Z的真实值。

  - 要求：$$\theta \to Z \to X$$满足马尔科夫性
  - 形式：已知$$P(z|\theta)$$求$$\arg \max_{\theta} \log P(x|\theta)$$

- 算法：

  - 假设一个$$\theta_n$$
  - 求出complete data的后验分布$$P(z|x, \theta_n)$$
  - 求出Log likelihood关于z的期望$$Q(\theta|\theta_n) = \int_z P(z|x,\theta_n) \log P(z|\theta)dz$$，以其极值点作为$$\theta_{n + 1}$$。

- 意义：可证明$$P(x|\theta_{n+1}) \ge P(x|\theta_{n})$$

  简化情况：

  - X为Z的一部分，即Z = (X, Y)，则Q函数可写为$$E_{Y|x, \theta_n} \log P(x, Y | \theta)$$
  - Z为n个随机变量z1, z2, ...组成，则Q函数可以相加

  引理：

  - 琴生不等式：若$$\phi$$是下凸函数，则$$E(\phi(x)) \ge \phi(E(x))$$（上凸相反）

  - 若$$Q(\theta|\theta_n) \ge Q(\theta_n|\theta_n)$$，则$$P(x|\theta) \ge P(x|\theta_n)$$
    $$
    \begin{aligned}
    \log P(x|\theta) & = \log\int_z P(z,x|\theta) dz \\
    	& = \log E_{z|x, \theta_n} \frac{P(z, x|\theta)}{P(z|x, \theta_n)} \\
    	& \ge E_{z|x, \theta_n} \log \frac{P(z, x|\theta)}{P(z|x, \theta_n)} \text{  (Jensen's inequality)} \\
    	& = E_{z|x, \theta_n} \log P(x|\theta_n) + E_{z|x, \theta_n} \log P(z|\theta) - E_{z|x, \theta_n} \log P(z|\theta_n) \\
    	& = \log P(x|\theta_n) + Q(\theta|\theta_n) - Q(\theta_n|\theta_n)
    \end{aligned}
    $$

  根据上式以及theta的迭代易得单调递增性。

- 在聚类中的运用：把类别信息看作缺失数据（上面的Y），用EM算法估计决定类别的参数和类别本身

  - 记号见28页
  - 用拉格朗日控制各类别的先验概率之和为1，然后优化
  - 通用混合模型分解算法：
    - 给出theta和先验概率P的初始值
    - 算类别的后验概率、log likelihood
    - 更新theta和P的估计
  - 解释k-means：分配类别即E步，重新计算聚类中心即M步
    - 类别的后验概率是一个退化分布（直接分给最近的）
    - M步的优化目标是使中心向量到类中所有点的平方距离和最小，也就是平均值



## 密度聚类与谱聚类

### DBSCAN

- 核心对象：其邻域内邻居数目超过某个阈值的点
- 边界对象：不是核心对象但在某个核心对象邻域内的点
- 可达性：
  - 直接密度可达：从p到q，q是p邻居，p是核心对象
  - 密度可达：p->p1->p2->q，中间都是直接密度可达
  - 密度相连：p和q都是从o密度可达的，则pq密度相连
- 聚类：
  - 密度相连是类性质，用于聚类
  - 聚类完全由其中的核心对象决定
  - 不属于任意非平凡类的点称为噪声
  - 算法：非常简单的深搜；如果用R*树可以到NlogN
  - 缺点：高维数据比较慢；不适合密度差异较大的数据；对参数特别敏感

### CDP

- 局部密度：也即邻域内邻居的个数，选取阈值使得平均局部密度为总个数的1%-2%
- 局部距离：如果当前样本的局部密度最大，则定义为所有对象与当前样本的距离最大值；否则定义为与所有局部密度大于当前局部密度（“更加中心”）的样本的距离最小值
- 聚类中心同时具有较大的局部密度和局部距离，可以以二者乘积为指标设定阈值确定聚类中心
- 其余点按照局部密度值由大到小依次分配到局部距离最小的类别（即与前面已经打好标的“更加中心”的点中最近的点相同）

### 谱聚类

- A为相似度矩阵（如邻接矩阵），需要对称
- $$\Delta$$为度矩阵（对角矩阵，第i位为第i点的度）
- 规范邻接矩阵$$M = \Delta^{-1}A$$（即把每个点的出度归一化）
- 图拉普拉斯矩阵$$L = \Delta - A$$
  - $$\begin{bmatrix}\sum_{j \ne 1} a_{1j} & - a_{12} & -a_{13} & \dots & -a_{1n} \\ -a_{21} & \sum_{j \ne 2} a_{2j} & -a_{23} & \dots & -a_{2n}  \\ \vdots & \vdots & \vdots & \ddots & \vdots \\  -a_{n1} & - a_{n2} & -a_{n3} & \dots & \sum_{j \ne n} a_{nj} \end{bmatrix}$$
  - 是对称半正定矩阵
  - 有点类似生灭过程的无穷小生成元（每行和是0）

- 规范对称拉普拉斯矩阵$$L_N = \Delta^{1/2}L\Delta^{-1/2}$$，即在每个位置除以两个节点的度的几何平均$$\sqrt{d_id_j}$$。
- 规范非对称拉普拉斯矩阵$$L_A = \Delta^{-1}L$$（即每一行除以该节点的度）。
- Ratio Cut聚类：
  - 最小化$$F_{RC}(C) = \sum_{i= 1}^k \frac{W(C_i,\bar{C_i})}{|C_i|} = \sum_{i= 1}^k \frac{c_i^{\top}Lc_i}{c_i^{\top}c_i}$$
  - 松弛后可令$$u_i = c_i / |c_i|$$，要求长度为1，用拉格朗日即可，可以发现优化结果为L除0外最小的k个特征值对应的特征向量，大致每行表征每个点属于每一类的概率
    - 也可以视为这张图在k维空间的谱嵌入
- 再把特征向量用kmeans聚类即可

- Normalized Cut：
  - 把|Ci|换成$$Vol(C_i) = W(C_i, V) = c_i^{\top} \Delta c_i$$
  - 可以发现最后是要最小化$$u^{\top}_i L_N u_i$$

- Average Cut:
  - 求类内平均相似度的最大值
  - 最后要最小化$$u^{\top}_i A u_i$$
- Min-Max Cut:
  - 最小化类间相似度与类内相似度的比



## 概率图模型

- 朴素贝叶斯：
  - MAP算法
  - 先验概率用样本集中某类所占比例来近似
  - 条件概率需要假设条件独立性，即给定类别内，维度与维度（特征与特征）之间独立（减少要估计的参数数目）
    - 同样用比例近似，但是在样本少时可能会出现0比较多的情况
    - Laplace smoothing：$$P(v_j|c_i) = \frac{n_{ij} + lp}{n_i + l}$$，其中p作为先验估计，l一般取为特征的数量
  - 如果是连续型数据，仍然按照维度进行参数估计（如假设为正态分布）
- 用有向无环图表示，结点为变量，边为因果关系
- 几种条件独立性
- 冲突结点：给定一条**无向**路径，其中X和Y是Z的两个邻居，且X和Y都影响Z
  - d分离和d连接（了解）



## 主题模型

### pLSA

- 文档-主题-单词的概率图模型

  - 单词与单词独立且不考虑顺序

- 用alpha和beta分别表示p(单词|主题)和p(主题|文档)，M和N分别代表文档和单词数，$$c(w_i, d_j)$$表示第i个词在第j篇文档中出现次数
  $$
  \max \sum_{j = 1}^M \sum_{i = 1}^N c(w_i, d_j) \log (\sum_{k = 1}^K \alpha_{ik}\beta_{kj})\\
  s.t.\ \sum_{i = 1}^N \alpha_{ik} = 1, \forall k ,\ \sum_{k = 1}^K \beta_{kj} = 1, \forall j
  $$

- 最大化产生文档集的概率

  - 不能直接求，因为log里面多个主题产生的概率相加

  - 利用下凸性变为k个log相加，并用拉格朗日求最大值以逼近，知权重恰好是主题的后验概率，并且这两个式子相等
    $$
    \log (\sum_{k = 1}^K \alpha_{ik} \beta_{kj}) = \log \sum_{k = 1}^K \sigma_k \frac{\alpha_{ik} \beta_{kj}}{\sigma_k} \ge \sum_{k = 1}^K \sigma_k \log \frac{P(w_i, t_k, d_j)}{\sigma_k} \\
    \sum_{k = 1}^K \sigma_k = 1 \\
    \max_{\sigma_k} \sum_{k = 1}^K \sigma_k \log \frac{P(w_i, t_k, d_j)}{\sigma_k} \to \sigma_k = P(t_k | w_i, d_j)
    $$

  - 定义$$Q(\theta^* | \theta) = \sum_{k = 1}^K P(t_k | w_i, d_j, \theta) \log \frac{P(w_i, t_k, d_j | \theta^*)}{P(t_k | w_i, d_j, \theta)}$$，类似EM算法迭代，则有：
    $$
    \log \sum_{k = 1}^K P(w_i, t_k, d_j | \theta_{t + 1}) = Q(\theta_{t+1} | \theta_{t + 1}) \ge Q(\theta_{t + 1} | \theta_t) = \max_{\theta^*} Q(\theta^* | \theta_t) \\
    \ge Q(\theta_t | \theta_t) = \log \sum_{k = 1}^K P(w_i, t_k, d_j | \theta_{t})
    $$
    其中第一个不等号的解释：改变第二个自变量相当于改变上面式子中的$$\sigma_k$$，而上面已经证明如此取的$$\sigma_k$$是上确界。

- 使用EM算法：

  - 希望最大化$$\sum_{j = 1}^M \sum_{i = 1}^N c(w_i, d_j) \log P(w_i, d_j | \theta)$$，可以把w、d视为observed data，t视为complete data的缺失部分，则可以使用EM算法：
    $$
    Q(\theta | \theta_n) = \sum_{j = 1}^M \sum_{i = 1}^N c(w_i, d_j) \sum_{k = 1}^K P(t_k|w_i, d_j, \theta_n) \log P(t_k, w_i, d_j | \theta)
    $$
    其中theta就是alpha和beta们。

  - 可得：
    $$
    Q(\theta | \theta_n) = \sum_{j = 1}^M \sum_{i = 1}^N c(w_i, d_j) \sum_{k = 1}^K \frac{\alpha_{ik} \beta_{kj}}{\sum_{k' = 1}^K \alpha_{ik'} \beta_{k'j}} (\log \hat{\alpha}_{ik} + \log \hat{\beta}_{kj}) + \\
     \sum_{j = 1}^M \sum_{i = 1}^N c(w_i, d_j) \sum_{k = 1}^K \frac{\alpha_{ik} \beta_{kj}}{\sum_{k' = 1}^K \alpha_{ik'} \beta_{k'j}} \log P(d_j | \theta)
    $$
    其中加hat的即theta，不加的即theta_n（都是常数！），另外由于文档的先验概率与参数无关，可知第二项是常数可忽略。令$$u_{ijk} = \frac{\alpha_{ik} \beta_{kj}}{\sum_{k' = 1}^K \alpha_{ik'}\beta_{k'j}} = P(t_k | w_i, d_j)$$并将alpha和beta的估计分开可得：
    $$
    L_1 = \sum_{j = 1}^M \sum_{i = 1}^N c(w_i, d_j) \sum_{k = 1}^K u_{ijk} \log \hat{\alpha}_{ik} - \sum_{j = 1}^M \lambda_j (\sum_{i = 1}^N \alpha_{ik} - 1) \\
    \frac{\partial L_1}{\partial \alpha_{ik}} = \sum_{j = 1}^M c(w_i, d_j) u_{ijk} / \hat{\alpha}_{ik} - \lambda_j = 0 ,\ \sum_{i = 1}^N \alpha_{ik} = 1\\
    \lambda_j = \sum_{j = 1}^M \sum_{i' = 1}^N c(w_{i'}, d_j) P(t_k | w_{i'}, d_j) \\
    \hat{\alpha}_{ik} = \frac{\sum_{j = 1}^M c(w_i, d_j) P(t_k | w_i, d_j)}{\sum_{j = 1}^M \sum_{i' = 1}^N c(w_{i'}, d_j) P(t_k | w_{i'}, d_j)}
    $$
    同理易得$$\hat{\beta}_{kj} = \frac{\sum_{i = 1}^N c(w_i, d_j) P(t_k | w_i, d_j)}{\sum_{k' = 1}^K \sum_{i = 1}^N c(w_{i}, d_j) P(t_{k'} | w_{i}, d_j)}$$。

### LDA

- 与LSA类似，但认为α、β（文档-主题和主题-单词的知识结构）是随机变量而不是确定的参数（贝叶斯观点）

  具体而言：

  - 用α、β两个参数向量（常量超参数，维度分别等于主题数和词汇数）生成狄利克雷分布（文档-主题分布和主题-单词分布的参数 的 先验分布，是多项分布的共轭分布（即后验也是狄利克雷分布，不用算积分））
  - 然后再从这两个分布中分别抽出文档-主题分布的参数θ和主题-单词分布的参数φ，
  - 最后再从θ和φ生成的多项分布中抽取主题和单词。

- 狄利克雷分布：一个 N维的1范数长度为1的向量 的分布， “分布的分布”

- 多项分布：类似二项分布的概念，但是有K种结果和K个概率参数，所以是一个K维向量每一维表示该结果出现次数。

- 主题的后验概率不容易直接计算（要积分），但是条件概率容易得到，故采用吉布斯采样。

  - $$\vec{z}_m$$表示第m个文档的所有单词的主题（V维向量），是隐变量
  - 先推导出主题和词的联合概率分布，可以推出恰好是两个狄利克雷分布相乘
  - 再由联合概率分布推导主题向量的条件分布
  - 用吉布斯采样得到后验分布

- 推导：符号见29页

  - 主题与词的联合分布：
    $$
    p(\vec{w}, \vec{z} | \vec{\alpha}, \vec{\beta}) = p(\vec{w}|\vec{z}, \vec{\beta})p(\vec{z}|\vec{\alpha}) \\
    
    \begin{aligned}
    
    p(\vec{w}|\vec{z}, \vec{\beta})
    	& = \int p(\vec{w}|\vec{z}, \bar\Phi)p(\bar\Phi|\vec{\beta}) d\bar\Phi \\
    	& = \int \prod_{k = 1}^K \prod_{t = 1}^V \phi_{k, t}^{n_{k,t}} (\prod_{k = 1}^K\frac{1}{\Delta(\vec{\beta})}\prod_{t = 1}^V \phi_{k, t}^{\beta_t - 1}) d\bar\Phi \\
    	& = \dots \\
    	& = \prod_{k = 1}^K \frac{\Delta(\vec{n}_k + \vec{\beta})}{\Delta(\vec{\beta})} \\
    p(\vec{z}|\vec{\alpha}) & = \int p(\vec{z}|\bar\Theta)p(\bar\Theta|\vec{\alpha}) d\bar\Theta \\
    	& = \dots \\
    	& = \prod_{d = 1}^M \frac{\Delta(\vec{m}_d + \vec{\alpha})}{\Delta(\vec{\alpha})} \\
    \end{aligned}
    $$
    其中$$\vec{n}_k$$表示所有词在主题k中出现的次数（合并所有文档），长度为V；$$\vec{m}_d$$表示所有主题在文档d中出现的次数（即属于每个主题的所有词出现的次数之和），长度为M。

  - 从联合分布推导z的条件分布（省略超参数alpha和beta）：
    $$
    \begin{aligned}
    p(z_i = k | \vec{z}_{\neg i}, \vec{w}) & \propto \frac{p(\vec{w} | \vec{z})}{p(\vec{w}_{\neg i} | \vec{z}_{\neg i})} \frac{p(\vec{z})}{p( \vec{z}_{\neg i})} \\
    	& \propto \frac{n^{(t)}_{k, \neg i} + \beta_t}{\sum_{v = 1}^V n^{(v)}_{k, \neg i} + \beta_v} (m_{d, \neg i}^{(k)} + \alpha_k)
    \end{aligned}
    $$
    其中$$w_i = t$$且它属于第d篇文档，$$n^{(v)}_{k, \neg i}$$表示去掉第i个单词后，单词v在主题k中出现的次数（合并所有文档），m类似。

  - 利用该条件分布采样得到z的后验分布

  - 利用z的后验分布得到theta和phi的后验分布，用其期望作为估计值：
    $$
    p(\vec{\theta}_d | \vec{z}_d, \vec\alpha) = Dir(\vec{\theta}_d | \vec\alpha + \vec{m}_d), E(\theta_{d,k}) = \frac{m_d^{(k)} + \alpha_k}{\sum_{l = 1}^K m_d^{(l)} + \alpha_l} \\
    p(\vec{\phi}_k | \vec{z}, \vec{w}, \vec\beta) = Dir(\vec{\phi}_k | \vec\beta + \vec{n}_k), E(\phi_{k,t}) = \frac{n_k^{(t)} + \beta_t}{\sum_{v = 1}^V n_k^{(v)} + \beta_v} \\
    $$



## 隐马尔可夫模型

- 观测值是状态的概率函数而不是状态本身

  - 比如，状态是天气（“晴”，“多云”，“雨”），但观测值是湿度（“高”，“中”，“低”）
  - 形式化定义为$$\lambda = (\pi, A, B)$$，其中π为初始状态概率分布（N维），A为转移概率矩阵（N * N），B为观测概率矩阵（N * M）。

- 基本问题：

  - 概率计算：给定模型和观测值，计算概率$$P(O|\lambda)$$
  - 预测：给定模型和观测值，计算最大似然的状态序列S
  - 学习：给定观测值，计算最大似然的模型λ

- 概率计算：

  - 穷举：
    $$
    \begin{aligned}
    P(O|\lambda) & = \sum_S P(O|S, \lambda)P(S|\lambda) \\
    	& = \sum_S \pi_{s_1}b_{s_1,o_1} \prod_{t = 2}^T a_{s_{t - 1}, s_t}b_{s_t, o_t}
    \end{aligned}
    $$
    复杂度为$$O(TN^T)$$。

  - 前向算法（动态规划）：

    令$$\alpha_t(i) = P(o_1, o_2, \dots, o_t, s_t = i|\lambda)$$为前向概率。则$$\alpha_{t+1}(j) = \sum_{i = 1}^N \alpha_t(i)a_{ij}b_{j, o_{t+1}}$$，复杂度为$$O(TN^2)$$

  - 后向算法：也是动归，顾名思义跟前向算法思路相反

- 预测算法：

  - 穷举法

  - 贪心法：每个时刻上都选择后验概率最大的状态。
    $$
    \hat{s}_t = \arg\max_{1 \le i \le N} \{ \gamma_t(i) \} \\
    \gamma_t(i) = P(s_t = i|O, \lambda) = \frac{P(O_{1:t}, s_t = i | \lambda)P(O_{(t+1):T}, s_t = i | \lambda)}{P(O|\lambda)} = \frac{\alpha_t(i)\beta_t(i)}{P(O|\lambda)}
    $$
    α和β是上面用动归计算的前向和后向概率。

  - Viterbi算法：

    从时刻t=1开始，递推计算在时刻t状态为i的各条部分路径的最大概率。
    $$
    \text{Let } \delta_t(i) = \max_{S_{1:(t-1)}}P(O_{1:t};S_{1:(t-1)}, s_t = i | \lambda) \\
    \delta_{t+1}(j) = \max_{1 \le i \le N} \delta_t(i)a_{ij}b_{j, o_{t+1}} \\
    \text{Let } \phi_{t+1}(j) = \arg\max_{1 \le i \le N} \delta_t(i)a_{ij}
    $$

- 学习算法：

  - 有监督（给出状态序列）：直接拟合转移概率即可

  - Baum-Welch算法：利用EM思想求局部最大概率

    - 确定似然函数形式$$\log P(O, S | \lambda)$$

    - $$Q(\lambda, \lambda_t) = \sum_S P(O,S | \lambda_t) \log P(O, S| \lambda)$$。（前面的概率与EM形式不同，但只是乘了一个与$$\lambda$$无关的常数$$P(O|\lambda_t)$$）

    - $$\lambda_{t+1} = \arg\max_{\lambda} Q(\lambda, \lambda_t)$$

    - 疯狂推导可得：
      $$
      \hat\pi_i = \frac{P(O, s_1 = i | \lambda)}{P(O|\lambda)} \\
      \hat{a}_{ij} = \frac{\sum_{t = 1}^{T - 1} P(O, s_t = i, s_{t+1} = j |\lambda)}{\sum_{t = 1}^{T - 1} P(O, s_t = i | \lambda)} \\
      \hat{b}_{jk} = \frac{\sum_{t = 1}^{T} P(O, s_t = j |\lambda)\delta_{t,k}}{\sum_{t = 1}^{T } P(O, s_t = j | \lambda)} \\
      \delta_{t, k} = \begin{cases}
      	1, & o_t = k \\
      	0, & o_t \ne k
      \end{cases}
      $$



## 条件随机场

- 在贝叶斯网络（有向图）的基础上，推广到无向图（只知道相关性）如马尔科夫网络，用边表示概率依存关系

  - 二者都属于概率图模型

- 马尔科夫随机场

  - $$P(X) = \frac{1}{Z}\prod_C \Psi_C(X_C)$$，C为图中的最大团（有很多个），$$\Psi$$为势函数，X代表团中所有变量，Z为归一化常数，称为配分函数
  - 条件独立性：假设存在x-w-y这样的结构，可以推出x和y在给定w后独立
  - 全局马尔科夫性质：如果上面的x/w/y都包括很多个变量，即任何从x中点到y中点的路径都必须经过w，仍有给定w，x与y独立，称为具有全局马尔科夫性质
  - HC定理：如果所有势函数都严格大于零，则马尔科夫随机场定义的所有概率分布都满足全局马尔科夫性
    - 因此势函数一般取$$e^{-E(X_C)}$$，其中E为能量函数

- 线性链条件随机场：

  - 能量函数表现为一系列特征函数的加权和，权重未知，特征已知

  - 给定X，求概率最大的Y

  - 概率计算：与HMM类似，主要是计算配分函数
    $$
    \begin{aligned}
    Z(x) & = \sum_{y_T}(\sum_{y_{1:T-1}}\Psi_T(y_T, y_{T - 1}, X)\prod_{t = 1}^{T - 1}\Psi_t(y_t, y_{t - 1}, X))\\
    	& = \sum_{j} \alpha_T(j) \\
    \end{aligned}
    $$
    使用前向或后向算法。

