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


# Fundamental of Higher Order Neural Networks for Modeling and Simulation [@guptaFundamentalsHigherOrder]

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

### Learning algortihm for second order neural units

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



# Nonconventional Neural Architectures and their Advantages for Technical Applications [@bukovskyNonconventionalNeuralArchitectures]

# Introduction #

- first mathematical model of neuron 1943
- principals for modeling of dynamic systems
	- customable non-linearity
	- order of dynamics of state space representation of a neuron
	- adaptable time delays

## HONU, HONN

- PNN - polynomial neural networks
- LNU, QNU, CNU
- linear optimization, avoidance of local minima

*bio-inspired* neuron, *perceptron*, *recurrent* (dynamic, hopfield)

static  vs dynamic \
continuous vs discrete
implementation of static/dynamic HONN


## Gradient optimization methods

- back propagation
- gradient descent rule
- Levenberg-Marquardt algorithm

## RHONN

- table page 14

**RTRL** -- real time recurrent learning

- dynamic version of gradient descent

**BPTT** -- back propagation throught time

- batch training technique can be implemented as combination of RTRL and L-M algorithm => RHONU

### Weighty update stability of static and dynamic version	


# Artificial High Order NN for Economics and Bussiness [@zhangArtificialHigherOrder] #

\newpage

# Bibliography

---
nocite: |
  @*
...
