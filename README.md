# Meta Learning ABC
An overview of the meta learning research area for beginners. :slightly_smiling_face: Thanks to the tutorial [Meta-Learning: from Few-Shot Learning to Rapid Reinforcement Learning](https://github.com/PKUCSS/MetaLearningABC/blob/master/slides/Meta%20Learning.pdf) by [Chelsea Finn](http://people.eecs.berkeley.edu/~cbfinn/) and [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)  and other references are listed in the middle of the tutorial.

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

![1568545357354](C:\Users\css\AppData\Roaming\Typora\typora-user-images\1568545357354.png) 

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





