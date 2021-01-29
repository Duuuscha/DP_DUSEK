---
title: Notes from bibliography
author: Roman Dušek
numbersections: true
geometry:
- top=30mm
- left=20mm
- right=20mm
- bottom=30mm
toc: true
language: cz
date: \today
---
\newpage

# Fundamental of Higher Order Neural Networks for Modeling and Simulation (Madan M. Gupta)

**Biological neuron**\
1. Synaptic operation
- strength (weight) is represented by previous knowledge. 

2. Somatic operation
- aggregation (summing), thresholding, nonlinear activation and dynamic processing
- output after certain threshold

if neuron was only linear the complex coginition would disappear

First neuron modeled (1943)
$$ u = \sum_{i=1}^n w_ix_i$$

## Higher Order Terms of Neural Inputs

year 1986, 1987, 1991, 1992, 1993
$$ u =\sum_{j=i}^n  \sum_{i=1}^n w_{ij}x_ix_j$$

## Activation functions

### Sigmoid

$$\phi(x)=\frac{1}{1+e^{-x}}$$

## SONU/QNU

- parameter reduction using upper triangular matrix of weights (XOR 6params vs 9 of 3 linear units)
$$u =\mathbf{x}^T_a\mathbf{W}_a\mathbf{x}_a=\sum_{j=i}^n  \sum_{i=1}^n w_{ij}x_ix_j$$
$$y = \phi(u)$$

### Learning (backpropagation)

the purpose of the neural units is to minimize the error E by adapting the weight
$$E(k)=\frac{1}{2}e(k)^2 \quad ; \quad e(k)=y(k) -y_d(k) $$

$$\mathbf{W}_a(k+1)=\mathbf{W}_a(k)+\Delta \mathbf{W}_a(k)$$
$$\Delta \mathbf{W}_a(k) =-\eta \frac{\delta E(k)}{\delta \mathbf{W}_a(k)}$$
where $\eta$ is lerning coefficient 
chain rule ...
# Nonconventional Neural Architectures and their Advantages for Technical Applications (Ivo Bukovsky)

- first mathematical model of neuron 1943
- principals for modeling of dynamic systems
	- customable non-linearity
	- order of dynamics of state space representaion of a neuron
	- adaptable time delays

## HONU, HONN

- PNN - polynomial neural networks
- LNU, QNU, CNU
- linear optimization, avoidance of local minima

*bio-inspired* neuron, *perceptron*, *recurent* (dynamic, hopfield)

static vs dynamic \
continous vs discrete
implementation of static/dynamic HONN


## Gradient optimization methods

- back propagation
- gradient descent rule
- Levenberg-Marquardt algorthm

## RHONN

- RTRL

## RTRL-real time recurrent learning

- dynamic version of gradient descent

## BPTT-back propagation throught time
- batch traning technique
- can be implemented as combination of RTRL and L-M algorithm => RHONU

