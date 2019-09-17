# Meta Learning ABC
An overview of the meta learning research area for beginners.    Thanks to the tutorial [Meta-Learning: from Few-Shot Learning to Rapid Reinforcement Learning](https://github.com/PKUCSS/MetaLearningABC/blob/master/slides/Meta%20Learning.pdf) by [Chelsea Finn](http://people.eecs.berkeley.edu/~cbfinn/) and [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)  and other references are listed in the middle of the tutorial.

Let's immerse ourselves in the world of meta learning :flags: . 

## Part 0: Popular Public Resources in Meta Learning 

Thanks to the curated list by [Sudharsan Ravichandiran](http://github.com/sudharsan13296) in the repo [Awesome-Meta-Learning](https://github.com/sudharsan13296/Awesome-Meta-Learning) 

### Lecture Videos 

* [Chelsea Finn: Building Unsupervised Versatile Agents with Meta-Learning](https://www.youtube.com/watch?v=i05Fk4ebMY0)

* [Sam Ritter: Meta-Learning to Make Smart Inferences from Small Data](https://www.youtube.com/watch?v=NpSpHlHpz6k)

* [Model Agnostic Meta Learning by Siavash Khodadadeh](https://www.youtube.com/watch?v=wT45v8sIMDM)

* [Meta Learning by Siraj Raval](https://www.youtube.com/watch?v=2z0ofe2lpz4)

* [Meta Learning by Hugo Larochelle](https://www.youtube.com/watch?v=lz0ekIVfoFs) 

* [Meta Learning and One-Shot Learning](https://www.youtube.com/watch?v=KUWywwvQv8E)


### Datasets

Most popularly used datasets:

* [Omniglot](https://github.com/brendenlake/omniglot) 
* [mini-ImageNet](https://github.com/y2l/mini-imagenet-tools) 
* [ILSVRC](http://image-net.org/challenges/LSVRC/)
* [FGVC aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
* [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)


### Workshops

* [MetaLearn 2017](http://metalearning.ml/2017/)
* [MetaLearn 2018](http://metalearning.ml/2018/)
* [MetaLearn 2019](http://metalearning.ml/2019/)


### Researchers

* [Chelsea Finn](http://people.eecs.berkeley.edu/~cbfinn/), _UC Berkeley_
* [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/), _UC Berkeley_
* [Erin Grant](https://people.eecs.berkeley.edu/~eringrant/),  _UC Berkeley_
* [Raia Hadsell](http://raiahadsell.com/index.html), _DeepMind_
* [Misha Denil](http://mdenil.com/), _DeepMind_
* [Adam Santoro](https://scholar.google.com/citations?hl=en&user=evIkDWoAAAAJ&view_op=list_works&sortby=pubdate), _DeepMind_
* [Sachin Ravi](http://www.cs.princeton.edu/~sachinr/), _Princeton University_
* [David Abel](https://david-abel.github.io/), _Brown University_
* [Brenden Lake](https://cims.nyu.edu/~brenden/), _Facebook AI Research_

### Popular Paper Reading Lists  

- https://github.com/sudharsan13296/Awesome-Meta-Learning 
- https://github.com/floodsung/Meta-Learning-Papers 

## Part 1 : Problem Statement

### 1.1 Aim: Learning to learn 

Meta-learning, also known as “learning to learn”, intends to **design models that can learn new skills or adapt to new environments rapidly with a few training examples.** 

![23609d09508ad9aeeda03b98abb1eb9.png](http://ww1.sinaimg.cn/large/006aGu85ly1g709jmb0d8j314o0kswhe.jpg)

It looks very similar to a normal learning task, but ***one dataset* is considered as *one data sample*.** 

<img src="https://lilianweng.github.io/lil-log/assets/images/few-shot-classification.png" style="zoom:50%;" /> 

### 1.2 How to train

> Our training procedure is based on a simple machine learning principle: test and train conditions must match”
> 															Vinyals et al., Matching Networks for One-Shot Learning

The general principle is to **reserve a test set for each task,so  we can learn meta parameters $\theta$ to generate specific model parameters $\phi$ rapidly to fit well for a new task.**     

<img src="http://ww1.sinaimg.cn/large/006aGu85ly1g709wl9h8zj312p0ivadc.jpg" alt="de5274732c9b498afe00865d2ee44d3.png" style="zoom:50%;" />

To cover it in one formula:

![222c1cd6c4ae005083e5bb1182c571c.png](http://ww1.sinaimg.cn/large/006aGu85ly1g709z8t5pdj30ii06rwes.jpg)

### 1.3 Closely related areas 

- **Multi-task Learning:** ![bbe7395ba00fa921d1ccb943036f100.png](http://ww1.sinaimg.cn/large/006aGu85ly1g70a35nperj30wp03cjru.jpg)

- **Hyper parameter Optimization**:  hyper parameters can be seen as $ \theta$,NN weights can be seen as $ \phi $  
- **Neural architecture  Search(NAS)**: architecture can be seen as $\theta$,NN weights can be seen as $\phi$ 

## Part 2: Overview: Meta Learning Algorithms

### General Recipe：

1. Choose a form of $p(\phi_i | D_i^{tr},\theta)$ 
2. Choose how to optimize $\theta$  w.r.t max-likehood objective using $D_{meta-train}$ 

- Black-box adaptation 
- Optimization-based inference 
- Non-parametric methods 
- Bayesian meta-learning 

### 2.1 Black-box Adaptation 

#### Definition

Train a neural network to represent $p(\phi_i|D_{i}^{tr},\theta)$，to simplify it ,use **deterministic** $\phi_i = f_{\theta}(D_i^{tr}) $  (non Bayesian way) 

![42dd2c45a6e29d61757e319eac5bef4.png](http://ww1.sinaimg.cn/large/006aGu85ly1g70b99j6z9j313r0au75r.jpg)

Possible forms of $f_{\theta}$ (all kinds of NN)： 

- LSTM 
- Neural turing machine (NTM) 
- 1-D convolutions 
- feed forward + average  

#### Challenges and Possible Solutions

Output all neural net parameters does not seem scalable. Idea: **Do not need to output all parameters of neural net, only sufficient statistics**. Please read the two papers listed below.

#### **Papers to read**

- [Meta-Learning with Memory-Augmented Neural Networks](http://proceedings.mlr.press/v48/santoro16.pdf) 
- [A SIMPLE NEURAL ATTENTIVE META-LEARNER](https://arxiv.org/pdf/1707.03141.pdf) 

### 2.2 Optimization-based inference  

#### Definition

Acquire $\phi_i$ through optimization: $ max_{\phi_i} log p (D_i^{tr}|\phi_i) + log p (\phi_i | \theta)$ 

 Meta-parameter $\theta$ serves as a prior,**e.g initialization for fine-tuning**  

![a1b2e2eae9be908fcf58256ff369fcb.png](http://ww1.sinaimg.cn/large/006aGu85ly1g70bxnh271j30m20870tc.jpg)

Other forms of prior:

- **explicit Gaussian prior:** [Meta-Learning with Implicit Gradients](https://arxiv.org/pdf/1909.04630.pdf) ![77d030e089094b99d1ee9d3cd637460.png](http://ww1.sinaimg.cn/large/006aGu85ly1g70duecgplj30hw02uq2w.jpg)
- **Bayesian linear regression on learned features**:  [Meta-Learning Priors for Efficient Online Bayesian Regression](https://arxiv.org/pdf/1807.08912.pdf)
- **Closed-form or convex optimization on learned features**:[Meta-learning with differentiable closed-form solvers](https://openreview.net/forum?id=HyxnZh0ct7), [Meta-Learning with Differentiable Convex Optimization](https://arxiv.org/pdf/1904.03758.pdf) 

#### Contrast: Optimization vs. Black-Box Adaptation

![](http://ww1.sinaimg.cn/large/006aGu85ly1g70c0cog3qj30l80a40tn.jpg)

#### Papers  to read

- [A Simple Neural Attentive Meta-Learner](https://arxiv.org/pdf/1707.03141.pdf)

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/pdf/1703.03400.pdf)

- [Meta-Learning and Universality: Deep Representations and Gradient Descent can Approximate any Learning Algorithm](https://arxiv.org/pdf/1710.11622.pdf) 

- [MAML approximates hierarchical Bayesian inference](https://arxiv.org/pdf/1801.08930.pdf) 

  

#### Challenges and Possible Solutions 

1. **How to choose architecture that is effective for inner gradient-step?** 

   idea： Progressive neural architecture search + MAML see: [Auto-Meta: Automated Gradient Based Meta Learner Search](https://arxiv.org/pdf/1806.06927.pdf) 

2. **Second-order meta-optimization can exhibit instabilities** 

   ![af18198d4b5944598ae562e36c20db4.png](http://ww1.sinaimg.cn/large/006aGu85ly1g70ecdahuoj30zz0bxab6.jpg)

  

### 2.3 Non-parametric methods  

#### Definition

**Use parametric meta-learners that produce effective non-parametric learners**.

![430bbb85bc7edad4c9d532f49042819.png](http://ww1.sinaimg.cn/large/006aGu85ly1g70eebbgesj30ym0framj.jpg)

See: [Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf) 

#### Challenges and possible solutions

**What if you need to reason about more complex relationships between data points?**

1. **Learn non-linear relation module on embeddings** : [Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/pdf/1711.06025.pdf) 
2. **Learn infinite mixture of prototypes** : [Infinite Mixture Prototypes for Few-Shot Learning](https://arxiv.org/pdf/1902.04552.pdf) 
3. **Perform message passing on embeddings**: [FEW-SHOT LEARNING WITH GRAPH NEURAL NETWORKS](https://arxiv.org/pdf/1711.04043.pdf)

### 2.4 Compare and Extend  

![59e96760f45281a9076ca1dceaf4796.png](http://ww1.sinaimg.cn/large/006aGu85ly1g70fn58yp8j31520crmyc.jpg)

### 2.5 Bayesian meta-learning  

#### Background 

**Few-shot learning problems may be ambiguous.** 

It's important for these situations:

1. safety-critical few-shot learning ( e.g. medical imaging)
2. Active learning:
   - [Active One-shot Learning](https://cs.stanford.edu/~woodward/papers/active_one_shot_learning_2016.pdf) 
   - [Learning Active Learning from Data](https://papers.nips.cc/paper/7010-learning-active-learning-from-data.pdf)
   - [Learning Algorithms for Active Learning](https://arxiv.org/pdf/1708.00088.pdf) 
3. learning to explore in meta-RL 

#### Key Idea 

![2b7c0d8f3c02579a61ea6edd9e0f86e.png](http://ww1.sinaimg.cn/large/006aGu85ly1g70fcxgvr0j30rz05lwey.jpg)

#### Papers to Read

- [Meta-Learning Probabilistic Inference For Prediction](https://arxiv.org/pdf/1805.09921.pdf) 
- [Amortized Bayesian Meta-Learning](https://openreview.net/forum?id=rkgpy3C5tX) 
- [Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm](https://arxiv.org/pdf/1608.04471.pdf) 
- [Bayesian Model-Agnostic Meta-Learning](https://arxiv.org/pdf/1806.03836)  
- [Modulating transfer between tasks in gradient-based meta-learning](https://openreview.net/forum?id=HyxpNnRcFX) 
- [Probabilistic Model-Agnostic Meta-Learning](https://arxiv.org/pdf/1806.02817.pdf) 

## Part 3: Meta-learning applications

Applications in computer vision :

- Few-shot image recognition
  - [Vinyals et al. Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf)
- Human motion and pose prediction
  - [Gui et al. Few-Shot Human Motion Prediction via Meta-Learning](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liangyan_Gui_Few-Shot_Human_Motion_ECCV_2018_paper.pdf)
  - [Alet et al. Modular Meta-Learning](https://arxiv.org/pdf/1806.10166.pdf)
- Domain adaptationcom 
  - [Li, Yang, Song, Hospedales. Learning to Generalize: Meta-Learning for Domain Adaptation](https://arxiv.org/pdf/1710.03463.pdf)
- Few-shot segmentation
  - [Shaban, Bansal, Liu, Essa, Boots. One-Shot Learning for Semantic Segmentation](https://www.cc.gatech.edu/~bboots3/files/OneShotSegmentation.pdf)
  - [Rakelly, Shelhamer, Darrell, Efros, Levine. Few-Shot Segmentation Propagation with Guided Networks](https://people.eecs.berkeley.edu/~rakelly/Rakelly_Shelhamer_revolver.pdf)
  - [Dong, Xing. Few-Shot Semantic Segmentation with Prototype Learning](http://bmvc2018.org/contents/papers/0255.pdf)

Applications in image & video generation :

- Few-shot image generation
  - [Reed, Chen, Paine, van den Oord, Eslami, Rezende, Vinyals, de Freitas. Few-Shot Autoregressive Density Estimation](https://arxiv.org/pdf/1710.10304.pdf)
- Few-shot image-to-image translation
  - [Liu, Huang, Mallya, Karras, Aila, Lehtinen, Kautz. Few-Shot Unsupervised Image-to-Image Translation](https://arxiv.org/pdf/1905.01723.pdf)
- Generation of novel viewpoints
  - [Gordon, Bronskill, Bauer, Nowozin, Turner. VERSA: Versatile and Efficient Few-Shot Learning](http://bayesiandeeplearning.org/2018/papers/10.pdf)
- Generating talking heads from images
  - [Zakharov, Shysheya, Burkov, Lempitsky. Few-Shot Adversarial Learning of Realistic Neural Talking Head Models](https://arxiv.org/pdf/1905.08233.pdf)

Applications in NLP :

- Adapting to new programs
  - [Neural Program Meta-Induction](https://arxiv.org/pdf/1710.04157.pdf)
  - [Natural Language to Structured Query Generation via Meta-Learning](https://arxiv.org/pdf/1803.02400.pdf)
- Adapting to new language
  - [Meta-Learning for Low-Resource Neural Machine Translation](https://aclweb.org/anthology/D18-1398)
- Learning new words
  - [Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf)
- Adapting to new personas
  - [Personalizing Dialogue Agents via Meta-Learning](https://arxiv.org/pdf/1905.10033.pdf)

Some more applications :

- One-hot **imitation** learning
  - [One-Shot Imitation Learning](https://arxiv.org/pdf/1703.07326.pdf)
  - [Task-Embedded Control Networks for Few-Shot Imitation Learning](https://arxiv.org/pdf/1810.03237.pdf)
  - [One-Shot High-Fidelity Imitation: Training Large-Scale Deep Nets with RL](https://arxiv.org/pdf/1810.05017.pdf)
  - [One-Shot Hierarchical Imitation Learning of Compound Visuomotor Tasks](https://arxiv.org/pdf/1810.11043.pdf)
- Learn from **weak** supervision
  - [One-Shot Imitation from Observing Humans via Domain-Adaptive Meta-Learning](https://arxiv.org/abs/1802.01557)
  - [Chelsea Finn - Berkeley bCourses](https://bcourses.berkeley.edu/courses/1468734/files/73079979/download?verifier=oYQ48sUcjC7k7N1MIk4si2Dshc40qxdPkoO1LXko&wrap=1)

## Part 4: Meta-reinforcement learning

Why should we care about meta-RL?

People can learn new skills **extremely** quickly, and we never learn from scratch. Maybe meta-RL algorithms can learn much more efficiently.

### Basic about RL

Markov decision process 										$M=\{S,A,P,r\}$

$S$ - state space 												states $s \in S$ (discrete or continuous)

$A$ - action space											  actions $a \in A$ (discrete or continuous)

$P$ - transition function									i.e. $p(s_{t+1}|a_t,s_t)=P(s_t,a_t,s_{t+1})$

$r$ - reward function 										 $r:S \times A \rightarrow \mathbb{R}$  eg: $r(s_t,a_t)$ - reward

$\pi_\theta(a|s)$ - policy with params $\theta$ 					  $\theta^\star=\arg \max_\theta E_{\pi_\theta}[\sum_{t=0}^Tr(s_t,a_t)]$



About the policy:

1. finite horizon :
   $$\theta^\star=\arg \max_\theta E_{\pi_\theta}[\sum_{t=0}^Tr(s_t,a_t)]$$
2. infinite horizon with discounted return :
   $$\theta^\star=\arg \max_\theta E_{\pi_\theta}[\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)]$$
3. stationary distribution :
   $$\theta^\star=\arg \max_\theta E_{\pi_\theta(s,a)}[r(s_t,a_t)]$$

consider $\pi_\theta(\tau)=p_\theta(s_1,a_1,...,s_T,a_T)=p(s_1)\prod_{t=1}^T\pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)$

We get $\theta^\star=\arg \max_\theta E_{\pi_\theta(\tau)}[R(\tau)]$

Here is the whole picture:

![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_aeafb2357a64bf578c146696e9e7d1f7.png)



### Review about Meta-learning

the dataset : $D_{meta-train}=\{(D_1^{tr},D_1^{ts}),...,(D_n^{tr},D_n^{ts})\}$

the training set : $D_i^{tr}=\{(x_1^i,y_1^i),...,(x_k^i,y_k^i)\}$

the test set :$D_i^{ts}=\{(x_1^i,y_1^i),...,(x_l^i,y_l^i)\}$

Meta-learning is tring to learn $\theta$ such that $\phi_i = f_\theta(D_i^{tr})$ is good for $D_i^{ts}$

**Probabilistic view :** 

$$\theta^\star = \arg \max_\theta \sum_{i=1}^n\log p (\phi_i|D_i^{ts}) \quad where\  \phi_i = f_\theta(D_i^{tr})$$

**Deterministic view :**

$$\theta^\star = \arg \min_\theta \sum_{i=1}^n \mathcal{L} (\phi_i,D_i^{ts}) \quad where\  \phi_i = f_\theta(D_i^{tr})$$





### The meta RL problem

Think about these four problems:

1. **The "Generic" learning (determinstic view)**
   $$\theta^\star = \arg \min_\theta \sum_{i=1}^n \mathcal{L} (\theta,D_i^{ts}) = f_{learn}(D^{tr})$$
2. **The "Generic" meta-learning (determinstic view)**
   $$ \theta^\star = \arg \min_\theta \sum_{i=1}^n \mathcal{L} (\phi_i,D_i^{ts}) \quad where\  \phi_i = f_\theta(D_i^{tr})$$
3. **Reinforcement learning**
   $$\theta^\star = \arg \max_\theta E_{\pi_\theta(\tau)}[R(\tau)]=f_{RL}(\mathcal{M}) \quad \mathcal{M} = \{S,A,P,r\}$$
4. **Meta-reinforcement learning**
   $$\theta^\star = \arg \max_\theta \sum_{i=1}^n E_{\pi_{\phi_i}(\tau)}[R(\tau)] \quad where \ \phi_i = f_\theta(\mathcal{M}_i)$$

Consider the meta-RL, at the training time, we assumpt $\mathcal{M}_i\sim p (\mathcal{M})$

At meta test-time, sample $\mathcal{M}_{test}\sim p(\mathcal{M})$, get $\phi_i = f_\theta(\mathcal{M}_{test})$



### Meta-RL with recurrent policies

Reconsider $\theta^\star = \arg \max_\theta \sum_{i=1}^n E_{\pi_{\phi_i}(\tau)}[R(\tau)] \quad where \ \phi_i = f_\theta(\mathcal{M}_i)$

The **main question** about meta-RL is how to implement $f_\theta(\mathcal{M}_i)$. 



What should $f_\theta(\mathcal{M}_i)$ do ?

1. improve policy with experience from $\mathcal{M}_i$ - $\{(s_1,a_1,s_2,r_1),...,(s_T,a_T,s_{T+1},r_T)\}$
2. choose how to interact, i.e. choose $a_t$

![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_25b20f30c4edc4e3b74de9b629eb69cc.png)


Actually, we just train a RNN policy!

$$ \theta^\star=\arg \max_\theta E_{\pi_\theta}[\sum_{t=0}^Tr(s_t,a_t)]$$

Optimizing total reward over the entire meta-episode with RNN policy automatically learns to explore!

### Architectures for meta-RL

- standard RNN(LSTM) architecture
![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_163ee26d6b62aa8900925eefea693320.png)
  [Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel. RL2: Fast Reinforcement Learning via Slow Reinforcement Learning. 2016.](https://arxiv.org/pdf/1611.02779.pdf)

- attention + temporal convolution

![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_f7473332fc0671e9921f4b082ad913d4.png)


  [Mishra, Rohaninejad, Chen, Abbeel. A Simple Neural Attentive Meta-Learner.](https://arxiv.org/pdf/1707.03141.pdf)

- parallel permutation-invariant context encoder

![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_7869d00416e5f2a4e818d7b8fa9aef2d.png)


  [Rakelly*, Zhou*, Quillen, Finn, Levine. Efficient Off-Policy Meta-Reinforcement learning via Probabilistic Context Variables](https://arxiv.org/pdf/1903.08254.pdf)

### MAML

Let's rethink the equation $$\theta^\star = \arg \max_\theta \sum_{i=1}^n E_{\pi_{\phi_i}(\tau)}[R(\tau)] \quad where \ \phi_i = f_\theta(\mathcal{M}_i)$$.

For standard RL problem, we can update the $\theta$ this way :

$$\theta^\star=\arg \max_\theta E_{\pi_\theta(\tau)}[R(\tau)]=\arg \max_\theta J(\theta)  \\ \theta^{k+1}\leftarrow \theta^k + \alpha \nabla_{\theta^k}J(\theta^k)$$

But if $f_\theta(\mathcal{M}_i)$ is an RL algorithm:

$f_\theta(\mathcal{M}_i) = \theta + \alpha \nabla_\theta J_i(\theta)$

we requires interacting with $\mathcal{M}_i$ to estimate $\nabla_\theta E_{\pi_\theta}[R(\tau)]$.

And this is model-agnostic meta-learning (MAML) for RL!



Picture for MAML:
 ![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/20181209221427217.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xpdWdsZW4=,size_16,color_FFFFFF,t_70)

Algorithm:

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/20181209221441670.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xpdWdsZW4=,size_16,color_FFFFFF,t_70)

For RL problem, we need to change the $\mathcal{L}$.

So we update $\theta$ use $\theta \leftarrow \theta + \beta \sum_i \nabla_\theta J_i[\theta + \alpha \nabla_\theta J_i(\theta)]$



Something more on MAML/gradient-based meta-learning for RL:

- Better MAML meta-policy gradient estimators: 
  - [Foerster, Farquhar, Al-Shedivat, Rocktaschel, Xing, Whiteson. DiCE: The Infinitely Differentiable Monte Carlo Estimator.](https://arxiv.org/pdf/1802.05098.pdf)
  - [Rothfuss, Lee, Clavera, Asfour, Abbeel. ProMP: Proximal Meta-Policy Search.](https://arxiv.org/pdf/1810.06784.pdf)
- Improving exploration: 
  - [Gupta, Mendonca, Liu, Abbeel, Levine. Meta-Reinforcement Learning of Structured Exploration Strategies.](https://arxiv.org/pdf/1802.07245.pdf)
  - [Stadie*, Yang*, Houthooft, Chen, Duan, Wu, Abbeel, Sutskever. Some Considerations on Learning to Explore via Meta-Reinforcement Learning.](https://arxiv.org/pdf/1803.01118.pdf)
- Hybrid algorithms (not necessarily gradient-based): 
  - [Houthooft, Chen, Isola, Stadie, Wolski, Ho, Abbeel. Evolved Policy Gradients.](https://arxiv.org/pdf/1802.04821.pdf)
  - [Fernando, Sygnowski, Osindero, Wang, Schaul, Teplyashin, Sprechmann, Pirtzel, Rusu. Meta-Learning by the Baldwin Effect.](https://arxiv.org/pdf/1806.07917.pdf)



### Meta-RL as partially observed RL

What's **partially observed** markov decision processes (POMDPs) ?

$\mathcal{M} = \{S,A,O,P,\epsilon,r \}$

$O$  - observation space 									observations $o \in O$ (discrete or continuous)

$\epsilon$ - emission probability $p(o_t|s_t)$ 

In $\pi_\theta(a|s,z)$, we think $z$ is the information policy needs to solve the current task. So learning a task is equivalent to inferring $z$ form the context $(s_1,a_1,s_2,r_1),...$ .

We can change the MDP to a POMDP !

$\tilde{\mathcal{M}} = \{\tilde S ,A,\tilde O,\tilde P,\epsilon, r\}$

$\tilde S = S \times Z$ 										$\tilde s = (s,z)$

$\tilde O = S$   											  $\tilde o = s$

So solving the POMDP $\tilde{\mathcal{M}}$ is equivalent to meta-learning.

To solve it, we need to estimate $p(s_t|o_{1:t})$ or $p(z_t|s_{1:t},a_{1:t},r_{1:t})$.

We can exploring via posterior sampling with latent context by doing the following two steps:

1. sample $z \sim \tilde p (z_t|s_{1:t},a_{1:t},r_{1:t})$
2. act according to $\pi_\theta(a|s,z)$ to collect more data

It is not optimal, but it's pretty good both in theory and in practice!






#### Perspectives on meta-RL

|                       | Advantages                                                   | Disadvantages                                                |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| RNN                   | -conceptually simple                                                       -relatively easy to apply | -vulnerable to meta-overfitting       -challenging to optimize in practice |
| Bi-level optimization | -good extrapolation (“consistent”)                                -conceptually elegant | -complex, requires many samples                              |
| Inference problem     | -simple, effective exploration via posterior sampling                     -elegant reduction to solving a special POMDP | -vulnerable to meta-overfitting   -challenging to optimize in practice |

#### Model-based meta-RL

Idea: 
- improve $\pi_\theta$ implicitly via model $\hat{p}(s_{t+1}|s_t,a_t)$

Advantages:
- requires much less data vs model-free
- a bit different due to model
- can adapt extremely quickly

Papers to read:

-[Saemundsson, Hofmann, Deisenroth. Meta-Reinforcement Learning with Latent Variable Gaussian Processes.](https://arxiv.org/abs/1803.07551?context=cs.LG)
-[Nagabandi, Finn, Levine. Deep Online Learning via Meta-Learning: Continual Adaptation for Model-Based RL.](https://arxiv.org/abs/1812.07671)
-[Nagabandi*, Clavera*, Liu, Fearing, Abbeel, Levine, Finn. Learning to Adapt in Dynamic, Real-World Environments Through Meta-Reinforcement Learning. ICLR 2019.](https://arxiv.org/abs/1803.11347v6)

## Challenges

### Meta-Overfitting

Cause of overfitting: 

- Meta learning requires task distributions, while specifying task distributions is hard when we have few meta-training tasks.

Possible Algorithms:

| Algorithm                      | Advantages                                                   | Disadvantages                                                |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| black-box adaptation           | simple and flexible models                                   | relies entirely on extrapolation of learned adaptation procedure |
| optimization-based             | at worst just gradient descent                               | pure gradient descent is not efficient without benefit of good initialization |
| non-parametric                 | at worst just nearest neighbor                               | does not adapt all parameters of metric on new data (might benearest neighbor in very bad space) |
| **unsupervised meta-learning** | solve tasks efficiently and don't need hand-specified labels | \                                                            |



Papers to read:

- Hsu, Levine, Finn. Unsupervised Learning via Meta-Learning. ICLR 2019](https://arxiv.org/abs/1810.02334?context=cs.CV)
  -[Gupta, Eysenbach, Finn, Levine. Unsupervised Meta-Learning for Reinforcement Learning.](http://arxiv.org/pdf/1806.04640)
  -[Eysenbach, Gupta, Ibarz, Levine. Diversity is All You Need.](https://arxiv.org/abs/1802.06070)
  -[Gupta, Eysenbach, Finn, Levine. Unsupervised Meta-Learning for Reinforcement Learning.](http://arxiv.org/pdf/1806.04640)
  -[Hsu, Levine, Finn. Unsupervised Learning via Meta-Learning.](http://arxiv.org/abs/1810.02334?context=stat.ML)
  -[Khodadadeh, Boloni, Shah. Unsupervised Meta-Learning for Few-Shot Image and Video Classification.](http://arxiv.org/abs/1811.11819)
  -[Metz, Maheswaranathan, Cheung, Sohl-Dickstein. Meta-Learning Update Rules for Unsupervised Representation Learning.](http://arxiv.org/abs/1804.00222)
  -[Ren, Triantafillou, Ravi, Snell, Swersky, Tenenbaum, Larochelle, Zemel. Meta-Learning for Semi-Supervised Few-Shot Classification.](http://arxiv.org/abs/1803.00676)

### Memorization

Problem: 

- If the task data isn’t strictly needed to learn the task, how to trade off information?

Possible way to solve:

- Provide demonstration and trials/language instruction/goal image/video tutorial.

Paper to read:

-[Zhou et al. Watch-Try-Learn: Meta-Learning Behavior from Demonstrations and Rewards, ‘19](http://arxiv.org/abs/1906.03352)	

## Ultimate Goal:Online Meta-Learning

Paper to read:

- [Finn*, Rajeswaran* et al. Online Meta-Learning ICML ‘19](http://arxiv.org/abs/1902.08438v1)