# Neural Ordinary Differential Equations for Langmuir Adsorption Kinetics

**Assignment Report**  
April 28, 2025

## Abstract

This report investigates the application of Neural Ordinary Differential Equations (Neural ODEs) to model Langmuir adsorption kinetics. We compare two approaches: a traditional parameter estimation method that directly estimates the rate constants in the Langmuir equation, and a Neural ODE model that learns the dynamics of the system without explicit knowledge of the underlying mathematical structure. Both methods are trained on synthetic data generated from the classic Langmuir adsorption model with added noise to simulate real-world measurements. Performance metrics including Root Mean Squared Error (RMSE), coefficient of determination (R²), and Mean Absolute Error (MAE) are evaluated to compare the methods' effectiveness. The results demonstrate that Neural ODEs can accurately model adsorption dynamics and compete with traditional parameter estimation approaches, highlighting their potential for modeling complex chemical systems where the exact mathematical form of the kinetics may be unknown.

## 1. Introduction

### 1.1 Langmuir Adsorption Kinetics

Langmuir adsorption is a fundamental model in surface chemistry that describes the adsorption of molecules onto a solid surface. The model makes several key assumptions:

1. The surface contains a fixed number of adsorption sites
2. Each site can hold at most one molecule of adsorbate (monolayer coverage)
3. All sites are energetically equivalent
4. There are no interactions between molecules on adjacent sites

The Langmuir model is described by the differential equation:

$$\frac{d\theta}{dt} = k_a P (1-\theta) - k_d \theta$$

Where:
- $\theta$ represents the fractional coverage of the surface (between 0 and 1)
- $k_a$ is the adsorption rate constant
- $k_d$ is the desorption rate constant
- $P$ is the pressure or concentration of the adsorbate in the gas or liquid phase

At equilibrium, the rate of adsorption equals the rate of desorption, leading to the classic Langmuir isotherm:

$$\theta_{eq} = \frac{K P}{1 + K P}$$

Where $K = \frac{k_a}{k_d}$ is the equilibrium constant.

### 1.2 Neural Ordinary Differential Equations

Neural Ordinary Differential Equations (Neural ODEs), introduced by Chen et al. in 2018, represent a novel paradigm in machine learning that bridges deep learning with differential equations. The core idea is to parameterize the derivative of the hidden state using a neural network:

$$\frac{dz(t)}{dt} = f_\theta(z(t), t)$$

Where $f_\theta$ is a neural network with parameters $\theta$.

Neural ODEs offer several advantages over traditional neural networks:
- Memory efficiency due to not requiring explicit layers
- Continuous depth rather than discrete layers
- Natural handling of irregular time series data
- Incorporation of physics-based inductive biases

For problems involving time series data governed by unknown differential equations, Neural ODEs can discover and represent the underlying dynamics without explicit knowledge of the mathematical form. This makes them particularly valuable for systems in chemical engineering where the exact governing equations might be complex or unknown.

### 1.3 Objectives

In this study, we investigate two approaches for modeling Langmuir adsorption kinetics:

1. **Parameter Estimation**: Directly estimate the rate parameters $k_a$ and $k_d$ in the Langmuir equation
2. **Neural ODE**: Learn the dynamics of the system without explicitly encoding the mathematical structure

Our goals are to:
- Implement and solve the Langmuir adsorption ODE in Julia
- Generate synthetic data from the model with added noise
- Train a Neural ODE to fit the synthetic data
- Compare the performance of the Neural ODE with traditional parameter estimation
- Evaluate the strengths and limitations of each approach
- Discuss implications for modeling complex adsorption systems

## 2. Methodology

### 2.1 Mathematical Framework

#### 2.1.1 Langmuir Adsorption Model

We implement the Langmuir adsorption model with the following parameters:
- True adsorption rate constant ($k_a$): 0.5
- True desorption rate constant ($k_d$): 0.1
- Pressure/concentration ($P$): 1.0 (constant)
- Initial surface coverage ($\theta_0$): 0.0

The governing equation is:

$$\frac{d\theta}{dt} = k_a P (1-\theta) - k_d \theta$$

This first-order ODE has an analytical solution:

$$\theta(t) = \frac{k