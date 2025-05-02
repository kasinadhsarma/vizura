# Neural ODEs Assignment - To-Do List

## 📋 Main Tasks Checklist

- [x] Implement and solve Langmuir adsorption ODE in Julia
- [x] Generate synthetic (ground truth) data from the ODE
- [x] Define and set up a Neural ODE model
- [x] Train Neural ODE to fit the generated data
- [x] Optimize model parameters (hyperparameter tuning)
- [x] Compare Neural ODE predictions with ground truth
- [x] Plot results and calculate error metrics
- [x] Write a detailed PDF report (Introduction, Methodology, Results, Conclusion)
- [x] Include plots and tables in the report
- [x] Upload report and code to Google Drive
- [x] Share Google Drive link in submission form

## 🛠 Setup Instructions

1. Install Julia (if not already installed):
   - Download from [Julia's official website](https://julialang.org/downloads/)
   - Add Julia to PATH during installation

2. Required Julia packages:
   ```julia
   using Pkg
   Pkg.add(["ComponentArrays", "Lux", "DiffEqFlux", "OrdinaryDiffEq", "Optimization", 
           "OptimizationOptimJL", "OptimizationOptimisers", "Random", "Plots"])
   ```

## 📊 Implementation Details

### Langmuir Adsorption ODE
- Implement the Langmuir adsorption differential equation:
  ```
  dθ/dt = ka(1-θ)C - kdθ
  ```
  where:
  - θ is the fractional surface coverage
  - ka is the adsorption rate constant
  - kd is the desorption rate constant
  - C is the concentration of adsorbate

### Neural ODE Model Development
1. Set up the neural network architecture
2. Define loss function comparing predictions with ground truth
3. Use different optimization algorithms (Adam, BFGS)
4. Analyze convergence and training stability

## 📚 Resources

- [Understanding Neural ODEs (Article)](https://jontysinai.github.io/jekyll/update/2019/01/18/understanding-neural-odes.html)
- [DiffEqFlux.jl Documentation](https://docs.sciml.ai/DiffEqFlux/stable/)
- [Julia's OrdinaryDiffEq.jl Documentation](https://docs.sciml.ai/DiffEqDocs/stable/)
- [Optimization.jl Documentation](https://docs.sciml.ai/Optimization/stable/)

## 📈 Progress Tracking

| Task | Status | Completion Date |
|------|--------|----------------|
| Setup Julia environment | completed | 26/04/2025 |
| Implement Langmuir ODE | completed | 27/04/2025 |
| Generate synthetic data | completed | 27/04/2025 |
| Define Neural ODE model | completed | 27/04/2025 |
| Train model | completed | 27/04/2025 |
| Optimize parameters | completed | 27/04/2025 |
| Compare results | completed | 27/04/2025 |
| Create visualizations | completed | 27/04/2025 |
| Write report | completed |02/05/2025|
| Submit assignment | completed |02/05/2025|

## 📅 Timeline

- Start date: April 27, 2025
- Submission deadline: May 2nd, 2025
- Intermediate milestones:
  - Model implementation: April 27, 2025 ✓
  - Training completion: April 27, 2025 ✓
  - Report draft: April 30, 2025
