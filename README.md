# Dynamical_System_Machine_Learning
We consider a quantity $x$ (a vector) which evolves with time, following a dynamical system. Think for example of the joint location of the planets in our solar system, which follows the law of gravitation.

Assume that, given an initial state $x(t=0) \in \mathbb{R}^n$ at time $t=0$, the time evolution of $x$ is governed by the following dynamical system:
$$\dot{x}(t) = f(x(t)) \quad \text(1) \quad \dot{x}(t) := \frac{dx(t)}{dt} $$
and  $f:\mathbb{R}^n \rightarrow \mathbb{R}^n$ is a given map describing the dynamics.

The goal of this practical session is to make use of some numerical solvers to improve the learning of dynamical systems with neural networks.
We first explore the case of discrete time and then the continuous time. 
