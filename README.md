#### Meta Learning ABC
An overview of the meta learning research area for beginners. Thanks to the tutorial [Meta-Learning: from Few-Shot Learning to Rapid Reinforcement Learning](https://github.com/PKUCSS/MetaLearningABC/blob/master/slides/Meta%20Learning.pdf) by [Chelsea Finn](http://people.eecs.berkeley.edu/~cbfinn/) and [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)  and other references are listed in the middle of the tutorial.

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

## Part 2: Meta Learning Algorithms

### General Recipe：

1. Choose a form of $p(\phi_i | D_i^{tr},\theta)$ 
2. Choose how to optimize $\theta$  w.r.t max-likehood objective using $D_{meta-train}$ 

- Black-box adaptation 
- Optimization-based inference 
- Non-parametric methods 
- Bayesian meta-learning 

### 2.1 Black-box Adaptation 

#### Definition

#### Papers to be read

### 2.2 Optimization-based inference  

#### Definition

#### Papers  to be read

### 2.3 Non-parametric methods  

#### Definition

#### Papers to be read

### 2.4 Bayesian meta-learning  

#### Definition

#### Papers to be read

### 2.5 Compare and Extend  