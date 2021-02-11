---
title: Notes from bibliography
author: Roman Dušek
numbersections: true
titlepage: true
geometry:
- top=30mm
- left=20mm
- right=20mm
- bottom=30mm
toc: true
language: en
date: \today
bibliography: bibliography.bib
colorlink: true
linkcolor: blue
citecolor: blue
urlcolor: blue
link-citations: true
---
\newpage


# Fundamental of Higher Order Neural Networks for Modeling and Simulation [@gupta2012]

## Introduction

**Biological neuron**\
1. Synaptic operation
- strength (weight) is represented by previous knowledge. 

2. Somatic operation
- aggregation (summing), thresholding, nonlinear activation and dynamic processing
- output after certain threshold

if neuron was only linear the complex cognition would disappear

First neuron modeled (1943)
$$ u = \sum_{i=1}^n w_ix_i$$

### Higher Order Terms of Neural Inputs

year 1986, 1987, 1991, 1992, 1993
$$ u =\sum_{j=i}^n  \sum_{i=1}^n w_{ij}x_ix_j$$

### Activation functions

#### Sigmoid

$$\phi(x)=\frac{1}{1+e^{-x}}$$

## SONU/QNU
 

- parameter reduction using upper triangular matrix of weights
$$u =\mathbf{x}^T_a\mathbf{W}_a\mathbf{x}_a=\sum_{j=i}^n  \sum_{i=1}^n w_{ij}x_ix_j$$
$$y = \phi(u)$$

if a weight is high it shows correlation between components of input patterns

### Learning algorithm for second order neural units

The purpose of the neural units is to minimize the error E by adapting the weight
$$E(k)=\frac{1}{2}e(k)^2 \quad ; \quad e(k)=y(k) -y_d(k) $$

$$\mathbf{W}_a(k+1)=\mathbf{W}_a(k)+\Delta \mathbf{W}_a(k)$$
$$\Delta \mathbf{W}_a(k) =-\eta \frac{\delta E(k)}{\delta \mathbf{W}_a(k)}$$
where $\eta$ is learning coefficient 
chain rule ...

using chain rule we get changes in the weight matrix as
$$\Delta \mathbf{W}_a (k) = -\eta e(k)\phi'(u(k))\mathbf{x}_a(k)\mathbf{x}_a^T(k)$$ 

Table with mathemathical structure and learning rule

| SONU   | Math. Struct           | Learning rule            |
|--------|:------------------------:|--------------------------|
| Static | $y_n=\mathbf{x_a^TWx}$ | Levenberg_marquard (L-M) |
|||Gradient descent|
| Dynamic | $y_n(k+n_s)=\mathbf{x_a^TWx}$ | Recurrent Gradient Descent  |
||| Backpropagation throughtime |

### Performance Assesment of SONU ###

#### XOR problem ####

- XOR 6params vs 9 of 3 linear units


## Time Series Prediction ##

## High order neural network units ##

- HONU is just a basic building block

### Example of Cubic neural network with two inputs ###

## Modified PNN ##

### Sigma-Pi NN ###

### Ridge PNN ###

## Conclusion ##




- this neural network first aggregates inputs and then multiplicate 



# Nonconventional Neural Architectures and their Advantages for Technical Applications [@bukovsky2012]

## Introduction

- first mathematical model of neuron 1943
- principals for modeling of dynamic systems
	- customable non-linearity
	- order of dynamics of state space representation of a neuron
	- adaptable time delays

### HONU, HONN

- PNN - polynomial neural networks
- LNU, QNU, CNU
- linear optimization, avoidance of local minima

*bio-inspired* neuron, *perceptron*, *recurrent* (dynamic, hopfield)

static  vs dynamic \
continuous vs discrete
implementation of static/dynamic HONN


### Gradient optimization methods

- back propagation
- gradient descent rule
- Levenberg-Marquardt algorithm

## RHONN

- table page 14

**RTRL** -- real time recurrent learning

- dynamic version of gradient descent

**BPTT** -- back propagation throught time

- batch training technique can be implemented as combination of RTRL and L-M algorithm => RHONU

## Weight update stability of static and dynamic version	


# Artificial High Order NN for Economics and Bussiness [@zhangArtificialHigherOrder2008] #


## Chapter 1
- use case of HONN
- model 1, 1b, 0
	- model 1 is containing one hidden layer with linear units
	- model 1b is containing two

**Polynomial HONN** uses poly-func as activation function

**Neural Adaptive HONN** uses adaptive functions as neuron

### Learning algorithm of HONN

### PHONN

### Trigonometric HONN
- uses trigonometric functions as activation functions

### Ultra high frequency cosine and sine HONN

### SINC and Sine Polynomial HONN

## Chapter 3

- HONN first introduced in (Giles, Maxwell 1987)
- hyperbolic tangent function
$$S(x)=\frac{e^x - e^{-x}}{e^{x}+e^{-x}}$$

## Chapter 5
- info about "High order Flexibel Neural Tree"

## Chapter 6
- most basic motivation of stock forecasting is financial gain
- motivation behind recurrent is that patterns may repeat in time

### Background HONN

### HONN structure

## Chapter 7
**Problems of ANNs**
- long convergence time
- can be stuck in local minima
- unable to handle high-frequency, non-linear and discontinuous data
- black box

**HONN**	can be considered as "open box"


## Chapter 8

### Introduction
- NN are data-driven we dont need prior assumptions
- NN can generalise
- they are universal approximators

**Well know problems with NNs**
- different results when tested on same datasets
- size sensitive, they suffer of over fitting

**Offline network** - goal of minimizing error over whole dataset\
**Online network** - aim is to adapt to local properties of the observed signal. They create detailed mapping of the underlying structure within the data

### Overview of NN
- using nonlinear transfer function they can carry out non-linear mappings
- history of milestone

#### Neuron structure


#### Activation Functions\
- threshold, linear, logistic sigmoid function

### Network structures
**Feed-forward**
-single layer (perceptron, ADALINE), multi layer
**RNN**
- using feedback loop\

**Fully recurrent**
	- all connections are trainable\

**Partial recurrent**
	- only feed-forward units are trainable, feedback utilize by *context unit*

- feedback results in nonlinear behavior, that provides networks with capabilities to storage information.

#### Lerning RNN

**Backpropagation through time**

- main idea is to unfold the RNN into an equivalent feed-forward network

**Real Time recurrent learning**

- each synaptic weight is updated for each representation of training set $\to$ no need to allocate memory proportional to the number of sequences

### HONN
(Giles & Maxwell, 1987)

- functional link network (Pao, 1989)

**Popular multi layer HONN**\

**Sigma-pi NN** (Rumelhar, Hinto & William, 1986)

- summing of inputs and product units (order is determined by number of inputs)

**Pi-Sigma NN** (Shin & Ghosh, 1992)

- summing of inputs and one product unit (fewer number of weights)

**Ridge polynomial neural network** (Shin & Ghost, 1991)

- using increasing number of pi-sigma units

#### High order interactions in Biological Networks

### Pipelined RNNs
(Haykin & Li, 1995)

- engineering principle - "divide and conquer"
- consist of two subsections - linear and nonlinear

#### RTRL for PRNN

### Second order PRNN

## Chapter 9 

### Second order Single layer RNN

## Chapter 10
- The lowest error of multilayer network occurs for one trained with Levenberg-Marquardt
- Multilayer networks have higher error


# Adaptive control with RHONN [@rovithakis2000]

## Introduction #

- for training model uses current and previous inputs, as well as the previous outputs
- in discrete outputs we need to discretized the model
- model is trained to identify the inverse dynamics of the plant instead of the forward dynamics
- by connecting the past neural output as input we make dynamic network that is highly nonlinear

**Problem with dynamic neural networks** that are based on static multilayer networks is that criterial functions possesses many local minima.

### Book goals ##

**Chapter 2** introduces RHONN\
**Chapter 3** online identification

## Identification using RHONN

### Model description

$$\dot{x}_i=-a_{i}x_{i}+b_{i} \left[ \sum_{k=1}^L w_{ik}z_k\right]$$ 

### Learning algorithms


# Discrete-time HONN [@sanchez2008]
- delete, not interesting

# Recurrent Neural Networks Tutotrial, Theano, NLP [@britz2015]

## Part 1

### RNN overview

- idea behind RNN is to make use of sequential information.
- we want to know what kind of information came before to deside output, this is something which we know as memory
	- this is limited only for looking few steps back
- we can unroll RNN to visualize the complete network
- RNN shares the same parameters across all steps

#### Training of RNN
- BPTT

### LSTM

- these are very good at capturing long-term dependencies using various types of gates

$x_t$ is input at time $t$\
$s_t$ is a hidden state\
input at current state is computated as $s_t = f(Ux_t+Ws_{t-1})$\
$o_{t}$ is output at time step $t$

## Part 2 [@britz2015]
- Building RNN class using numpy
- calculation loss using *cross-entropy*
- training model using BPTT
- gradient checking

## Part 3 [@britz2015b]

- BPTT
- Vanishing Gradient Problem
	- problem of learning long-range dependencies-
	- proper initialization of W helps, regularization, ReLU instead of tanh
	- using LSTM and GRU
- Exploding gradient problem
	- clipping weights when they get too high

## Part 4 [@britz2015c]

- implementing LSTM
	- $i, f, o$ are called input, forget and output gate. Called gate because the sigmoid function squashes the value between 0/1. By multiplying them with vector you define how much of that other vector you let through. 
		- Input gate defines how much of the newly computed state for the current input will be let through.
		- Fogrget gate defines how much of the previous state is gonna be kept.
		- Output gate defines how much of the external state is gonna be exposed to the external network
	- $g$ je a candidat hidden state, computed based on current and previous hidden state.
	- $c_t$ is internal memory of the unit, combined of last internal memory $c_{t-1}$ multiplied with forget gate and newly computed hidden state.
	- $s_t$ is output hidden state computed by multiplying memory $c_t$ with output gate

$$i =\sigma(x_tU^i + s_{t-1} W^i) \\ f =\sigma(x_t U^f +s_{t-1} W^f) \\ o =\sigma(x_t U^o + s_{t-1} W^o) \\ g =\ tanh(x_t U^g + s_{t-1}W^g) \\ c_t = c_{t-1} \circ f + g \circ i \\ s_t =\tanh(c_t) \circ o $$

- implementing GRU, code in Theano
	- $r$ reset gate, determines, how to combine the new input with the previous memor, and the update gate
	- $z$ update gate, defines how much of the previous memory to keep around.
	 

$$ z =\sigma(x_tU^z + s_{t-1} W^z) \\ r =\sigma(x_t U^r +s_{t-1} W^r) \\ h = tanh(x_t U^h + (s_{t-1} \circ r) W^h) \\ s_t = (1 - z) \circ h + z \circ s_{t-1} $$

# Recurrent neural network for prediction [@mandic2001]

## Introduction

- history about neural networks
- structure of neural network
	- performance achieved via dense interconnection of simple computational elements
	- interconnected neural provide robustness
- perspective on time series prediction
	- Yule 1927, introduction of autoregressive model
		- AR, MA and ARMA are linear
	- high computational complexity of RNN ($\mathcal{O}(N^4)$)
	- Pros of using RNN from time series prediction
		- are fast, if not faster than most available statistical techniques
		- self monitoring
		- accurate if not accurate as most s. t.
		- iterative forecasts
		- they cope with non-linearity and non-stationary
		- parametric and non-parametric prediction
	- Most difficult systems to predict
		- non-stationary dynamics, time-variant (speech)
		- noise, experiment error (bio-signals)
		- dealing with short time series (heart rate)
	- two IEEE dedicated issues of ANN for signal processing

## Fundamentals

**Adaptability** is ability to react in sympathy with disturbance
- system learned by supervised learning is adaptive
**Gradient calculation**\

**Leaning methodology**
- deterministic, stochastic, adaptive,

**Batch vs. incremental**

**Curse of dimensionality**

**Transformation on the input data**\
- normalisation - dividing each input by it's squared norm
- Rescaling - multiplying/adding constant
- standardisation - around 0 (-1 to 1)
	- helps with change of weights
- Principal component analysis

Nonlinear transformation as $log()$

## Network Architectures for Prediction

- commonality of adaptive filters and RNN

## Activation functions used in Neural Networks

- mostly sigmoid informatio

## RNN architecture


# 30 years of adaptive neural network [@widrow1990]

## Nonlinear classifier
- using polynomial preprocessor


\newpage

# Bibliography

---
nocite: |
  @*
...
