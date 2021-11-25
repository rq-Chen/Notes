# Machine Learning

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
  - Pre-training:
    - Mask Language Modeling (MLM): essentially the *Cloze* task (完形填空), randomly masking some words in the input and predicting them
    - Next Sentence Prediction (NSP): given a pair of sentence, predicting whether they are consecutive
    - Datasets: *BooksCorpus* (800M words) and English Wikipedia (2500M words)\
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

      - the guardian asked GPT-3 to write an [essay](https://www.theguardian.com/commentisfree/2020/sep/08/robot-wrote-this-article-gpt-3) to prove that it is not harmful to human, given an introduction paragraph; the outputs are quite good and comparable to human
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