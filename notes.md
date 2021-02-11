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
$o_{t}$ is output at time step $t$\





\newpage

# Bibliography

---
nocite: |
  @*
...
