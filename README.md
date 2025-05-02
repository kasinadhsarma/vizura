# Neural ODEs for Langmuir Adsorption Kinetics

This project implements Neural Ordinary Differential Equations (Neural ODEs) to model Langmuir adsorption kinetics. The implementation compares traditional parameter estimation with Neural ODE approaches.

## 🎯 Project Overview

The project demonstrates the application of Neural ODEs to model and predict Langmuir adsorption behavior, comparing the results with traditional ODE solutions.

### Key Features
- Implementation of Langmuir adsorption ODEs
- Neural ODE model development and training
- Parameter estimation using both traditional and Neural ODE approaches
- Comprehensive error metrics and visualization
- Modular codebase with improved implementation

## 📋 Requirements

- Julia 1.8 or higher
- Required Julia packages:
```julia
using Pkg
Pkg.add([
    "DifferentialEquations",
    "Plots",
    "Flux",
    "ComponentArrays",
    "Optimization",
    "OptimizationOptimJL",
    "OptimizationOptimisers",
    "DiffEqFlux",
    "Zygote",
    "Random",
    "Statistics",
    "BSON"
])
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/kasinadhsarma/vizura.git
cd vizura
```

2. Install dependencies:
```julia
julia --project -e 'using Pkg; Pkg.instantiate()'
```

3. Run the examples:
```julia
julia --project NeuralODEs.jl    # Basic Neural ODE example
julia --project LangmuirODE.jl   # Langmuir adsorption implementation
julia --project improved.jl       # Improved implementation with better organization
```

## 📁 Project Structure

```
vizura/
├── NeuralODEs.jl          # Basic Neural ODE implementation
├── LangmuirODE.jl         # Langmuir adsorption kinetics
├── improved.jl            # Improved implementation
├── README.md             # This file
└── LICENSE               # License file
```

## 📊 Implementation Details

### Langmuir Adsorption Model
The project implements the Langmuir adsorption differential equation:
```
dθ/dt = ka(1-θ)C - kdθ
```
where:
- θ: fractional surface coverage
- ka: adsorption rate constant
- kd: desorption rate constant
- C: concentration of adsorbate

### Neural ODE Implementation
- Uses DiffEqFlux.jl for Neural ODE implementation
- Implements both traditional parameter estimation and Neural ODE approaches
- Includes comprehensive error metrics (RMSE, R², MAE)
- Provides visualization tools for results comparison

## 📈 Results

The implementation generates:
- Comparison plots between true and predicted solutions
- Training loss curves
- Error metrics for both approaches
- Parameter estimation results

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- [Neural Ordinary Differential Equations Paper](https://arxiv.org/abs/1806.07366)
- [DiffEqFlux.jl Documentation](https://docs.sciml.ai/DiffEqFlux/stable/)
- [Julia SciML Documentation](https://docs.sciml.ai/)

## ✍️ Authors

- Kasinadh Sarma - Initial work and implementation

## 📧 Contact

For questions and feedback, please open an issue in the repository.