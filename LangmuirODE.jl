# Import necessary packages
using DifferentialEquations
using Plots
using OrdinaryDiffEq
using Flux
using ComponentArrays # Useful for handling nested parameters, though destructure also works
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using DiffEqFlux
using Zygote
using Random # For reproducibility
using Statistics # For error metrics calculation

# Set random seed for reproducibility
Random.seed!(123)

# --- Error Metrics Functions ---
# Root Mean Square Error (RMSE)
function rmse(predictions, targets)
    sqrt(sum((predictions .- targets).^2) / length(predictions))
end

# Coefficient of determination (R²)
function r_squared(predictions, targets)
    ss_res = sum((targets .- predictions).^2)
    ss_tot = sum((targets .- mean(targets)).^2)
    return 1 - ss_res / ss_tot
end

# Mean Absolute Error (MAE)
function mae(predictions, targets)
    mean(abs.(predictions .- targets))
end

# Define the Langmuir adsorption ODE model
# θ is the fractional surface coverage
# dθ/dt = ka * P * (1-θ) - kd * θ
# where:
# - ka is the adsorption rate constant
# - kd is the desorption rate constant
# - P is the pressure of the gas phase (assumed constant)

# True parameters
const ka_true = 0.5   # adsorption rate constant
const kd_true = 0.1   # desorption rate constant
const P = 1.0         # pressure (constant)

# Langmuir ODE function with the true parameters
function langmuir_ode!(dθ, θ, p, t)
    # p is not used here, using global constants
    dθ[1] = ka_true * P * (1 - θ[1]) - kd_true * θ[1]
end

# Initial condition (use Float32 for consistency with NN)
θ0 = Float32[0.0]
tspan = (0.0f0, 20.0f0) # Use Float32 for time span as well

# Create the ODE problem with true parameters
prob_true = ODEProblem(langmuir_ode!, θ0, tspan)

# Solve the ODE problem to generate synthetic data
sol_true = solve(prob_true, Tsit5(), saveat=0.5f0)

# Add some noise to the data to simulate real measurements
noise_level = 0.01f0
# Ensure noisy_data is Float32
noisy_data = Array{Float32}(sol_true) + noise_level * randn(Float32, size(Array(sol_true)))
# Get time points as Float32
t_points = Float32.(sol_true.t)

# Plot the true solution and the noisy data
plt = plot(t_points, Array(sol_true)[1,:], label="True Solution",
           title="Langmuir Adsorption Model",
           xlabel="Time", ylabel="Surface Coverage (θ)",
           legend=:bottomright, lw=2)
scatter!(plt, t_points, noisy_data[1,:], label="Noisy Data", alpha=0.5)
# display(plt) # Displaying plots might not work in all environments, saving is safer

# --- Neural ODE Part ---

# Define a neural network that represents the ODE function
# Ensure layers output Float32
nn_architecture = Flux.Chain(
    Flux.Dense(1 => 10, tanh; init=Flux.glorot_uniform(gain=1.0f0), bias=true),
    Flux.Dense(10 => 10, tanh; init=Flux.glorot_uniform(gain=1.0f0), bias=true),
    Flux.Dense(10 => 1; init=Flux.glorot_uniform(gain=1.0f0), bias=true)
)

# Get initial parameters and the restructure function
p_nn_init, re = Flux.destructure(nn_architecture)
# Ensure parameters are Float32
p_nn_init = Float32.(p_nn_init)

# Define the neural ODE *dynamics* function correctly
# It MUST use the parameters `p` passed by the solver/optimizer
function nn_dynamics!(dθ, θ, p, t)
    # Restructure the flat parameter vector `p` into the network
    nn_local = re(p)
    # Network prediction (ensure input type matches parameters, should be Float32)
    nn_output = nn_local(θ)
    # Assign the output to the derivative
    # Use .= for in-place modification if dθ is mutable
    # If dθ is Number (like in this 1D case), just return
    # Check the type/structure of dθ expected by the solver for the specific problem
    # For a 1D problem where θ is Vector{Float32}, dθ is also Vector{Float32}
    dθ .= nn_output
end

# Create the Neural ODE Problem stub (parameters will be updated in predict)
# Pass the initial parameters p_nn_init here
nn_prob = ODEProblem{true}(nn_dynamics!, θ0, tspan, p_nn_init) # Use Float32 initial params

# Prediction function using current parameters `p`
function predict_neural(p)
    # Create a new problem with the current parameters `p`
    # The state u0 and timespan tspan are fixed
    _prob = remake(nn_prob, p=p)
    # Solve the ODE problem. Ensure solver tolerances and types match Float32 if needed.
    # Use saveat=t_points to match the data points.
    sol = solve(_prob, Tsit5(), saveat=t_points, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    # Check if solve was successful
    if sol.retcode != ReturnCode.Success
        # Return something that leads to high loss, e.g., NaNs or Infs
        # Be careful with type stability if returning different types
        return fill(Float32(Inf), size(noisy_data))
    end
    # Return the solution array, ensuring it's Float32
    return Array{Float32}(sol)
end

# Loss function: Mean squared error
# Takes parameters `p` and an unused placeholder `_` (often for data batches)
function loss_neural(p, _)
    pred = predict_neural(p)
    # Handle potential Inf values from failed solves
    if any(isinf, pred)
        return Float32(Inf)
    end
    # Calculate loss, ensure types match
    sum(abs2, pred .- noisy_data) / length(noisy_data)
end

# Callback function to monitor training progress
iterations = Ref(0)
callback = function (p, l)
    iterations[] += 1
    if iterations[] % 10 == 0 || iterations[] == 1
        println("Iteration: $(iterations[]), Loss: $l")
        # Optional: Check for NaN/Inf loss
        if isnan(l) || isinf(l)
            println("Loss is NaN or Inf, stopping training.")
            return true # Return true to stop optimization
        end
    end
    return false # Return false to continue optimization
end

# Train the neural ODE using OptimizationOptimisers
# Define the OptimizationFunction with the correct loss and AD backend
optf_neural = OptimizationFunction(loss_neural, Optimization.AutoZygote())

# Define the OptimizationProblem with the function, initial parameters, and placeholder
optprob_neural = OptimizationProblem(optf_neural, p_nn_init, nothing) # Pass nothing as the placeholder argument

# Solve the optimization problem
# Use ADAM optimizer, specify learning rate (e.g., 0.01)
# Set maxiters for the number of training iterations
println("Starting Neural ODE training...")
result_neural = solve(optprob_neural, ADAM(0.01f0), callback=callback, maxiters=300)
println("Neural ODE training finished.")

# Get final trained parameters
final_params_neural = result_neural.u

# Predict with the trained model
nn_sol_final = predict_neural(final_params_neural)

# Plot results
plt_result = plot(t_points, Array(sol_true)[1,:], label="True Solution",
                 title="Neural ODE vs True Dynamics",
                 xlabel="Time", ylabel="Surface Coverage (θ)",
                 legend=:bottomright, lw=2)
scatter!(plt_result, t_points, noisy_data[1,:], label="Noisy Data", alpha=0.5)
# Check if the final prediction is valid before plotting
if !any(isinf, nn_sol_final)
    plot!(plt_result, t_points, nn_sol_final[1,:], label="Neural ODE", ls=:dash, lw=2)
else
    println("Neural ODE prediction contains Inf/NaN, not plotting.")
end
# display(plt_result)
savefig(plt_result, "langmuir_neural_ode_result.png")
println("Neural ODE result plot saved to langmuir_neural_ode_result.png")


# --- Parameter Estimation Part ---
# (Should work mostly as before, but ensure type consistency if needed)
i in 1:size(table_data, 1)i in 1:i in 1:size(table_data, 1) in 1:size(table_data, 1)e(table_data, 1)
println("\nStarting Parameter Estimation...")

# Define the Langmuir ODE function with parameters to be learned
# Ensure types are handled correctly if p is Float32
function langmuir_ode_param!(dθ, θ, p, t)
    ka, kd = p
    # Promoti in 1:size(table_data, 1)e P and constants if needed, though P is Float64 here.
    # It's generally better if everything is the same type.
    # Let's redefine P as Float32 for consistency.
    P_f32 = Float32(P)
    dθ[1] = ka * P_f32 * (1 - θ[1]) - kd * θ[1]
end

# Initial guess for parameters (use Float32)
p_initial_param = Float32[0.3, 0.2]
# Create ODE problem for parameter estimation with Float32 types
param_prob = ODEProblem{true}(langmuir_ode_param!, θ0, tspan, p_initial_param)

# Prediction function for parameter estimation
function predict_param(p)
    _prob = remake(param_prob, p=p)
    # Use same solver settings and save points
    sol = solve(_prob, Tsit5(), saveat=t_points)
    if sol.retcode != ReturnCode.Success
        return fill(Float32(Inf), size(noisy_data))
    end
    Array{Float32}(sol)
end

# Loss function for parameter optimization
function loss_param(p, _)
    pred = predict_param(p)
    if any(isinf, pred)
        return Float32(Inf)
    end
    sum(abs2, pred .- noisy_data) / length(noisy_data)
end

# Reset iterations counter for the callback
iterations[] = 0

# Train the parametric model using OptimizationOptimJL (BFGS is often good for smaller param numbers)
optf_param = OptimizationFunction(loss_param, Optimization.AutoZygote()) # Or specify another AD
optprob_param = OptimizationProblem(optf_param, p_initial_param, nothing)

# Note: BFGS (from OptimJL) might expect Float64 parameters.
# If using BFGS causes issues with Float32, either:
# 1. Switch back to Float64 for the parameter estimation part.
# 2. Use an optimizer from OptimizationOptimisers (like ADAM) which handles Float32 better.
# Let's try ADAM first for consistency.
# result_param = solve(optprob_param, BFGS(initial_stepnorm=0.01), callback=callback, maxiters=100)
println("Training parametric model with ADAM...")
result_param = solve(optprob_param, ADAM(0.01f0), callback=callback, maxiters=300)
# If ADAM doesn't converge well, BFGS might be better but may require Float64 setup.
# Example with BFGS (might need Float64 setup):
# θ0_f64 = Float64[0.0]; tspan_f64 = (0.0, 20.0); noisy_data_f64 = Float64.(noisy_data); t_points_f64 = Float64.(t_points)
# p_initial_param_f64 = Float64[0.3, 0.2]
# param_prob_f64 = ODEProblem(...) # setup with f64
# predict_param_f64(p) ... solve(...saveat=t_points_f64) ... Array{Float64}(sol)
# loss_param_f64(p, _) ... predict_param_f64(p) .- noisy_data_f64 ...
# optf_param_f64 = OptimizationFunction(loss_param_f64, Optimization.AutoZygote())
# optprob_param_f64 = OptimizationProblem(optf_param_f64, p_initial_param_f64, nothing)
# result_param = solve(optprob_param_f64, BFGS(), callback=callback, maxiters=100)


println("Parameter estimation training finished.")

# Get the learned parameters
learned_params = result_param.u
println("True parameters: ka = $ka_true, kd = $kd_true")
println("Learned parameters: ka = $(learned_params[1]), kd = $(learned_params[2])")

# Final prediction with learned parameters
param_sol_final = predict_param(learned_params)

# Plot comparison
plt_comparison = plot(t_points, Array(sol_true)[1,:], label="True Solution",
                     title="Parameter Learning vs Neural ODE",
                     xlabel="Time", ylabel="Surface Coverage (θ)",
                     legend=:bottomright, lw=2)
scatter!(plt_comparison, t_points, noisy_data[1,:], label="Noisy Data", alpha=0.5)
if !any(isinf, nn_sol_final)
    plot!(plt_comparison, t_points, nn_sol_final[1,:], label="Neural ODE", ls=:dash, lw=2)
end
if !any(isinf, param_sol_final)
    plot!(plt_comparison, t_points, param_sol_final[1,:], label="Learned Parameters", ls=:dot, lw=2)
else
    println("Parametric prediction contains Inf/NaN, not plotting.")
end
# display(plt_comparison)
savefig(plt_comparison, "langmuir_comparison.png")
println("Comparison plot saved to langmuir_comparison.png")

# Calculate and print error metrics
println("\n=== ERROR METRICS ===")

# Get true solution (without noise) for comparison
true_solution = Array(sol_true)[1,:]

# Neural ODE metrics (comparing to true solution)
if !any(isinf, nn_sol_final)
    neural_rmse = rmse(nn_sol_final[1,:], true_solution)
    neural_r2 = r_squared(nn_sol_final[1,:], true_solution)
    neural_mae = mae(nn_sol_final[1,:], true_solution)
    
    println("\nNeural ODE vs True Solution:")
    println("RMSE: $(round(neural_rmse, digits=5))")
    println("R²: $(round(neural_r2, digits=5))")
    println("MAE: $(round(neural_mae, digits=5))")
    
    # Neural ODE vs noisy data
    neural_rmse_noisy = rmse(nn_sol_final[1,:], noisy_data[1,:])
    neural_r2_noisy = r_squared(nn_sol_final[1,:], noisy_data[1,:])
    neural_mae_noisy = mae(nn_sol_final[1,:], noisy_data[1,:])
    
    println("\nNeural ODE vs Noisy Data:")
    println("RMSE: $(round(neural_rmse_noisy, digits=5))")
    println("R²: $(round(neural_r2_noisy, digits=5))")
    println("MAE: $(round(neural_mae_noisy, digits=5))")
else
    println("\nNeural ODE metrics unavailable due to invalid solution")
end

# Parameter estimation metrics (comparing to true solution)
if !any(isinf, param_sol_final)
    param_rmse = rmse(param_sol_final[1,:], true_solution)
    param_r2 = r_squared(param_sol_final[1,:], true_solution)
    param_mae = mae(param_sol_final[1,:], true_solution)
    
    println("\nParameter Estimation vs True Solution:")
    println("RMSE: $(round(param_rmse, digits=5))")
    println("R²: $(round(param_r2, digits=5))")
    println("MAE: $(round(param_mae, digits=5))")
    
    # Parameter estimation vs noisy data
    param_rmse_noisy = rmse(param_sol_final[1,:], noisy_data[1,:])
    param_r2_noisy = r_squared(param_sol_final[1,:], noisy_data[1,:])
    param_mae_noisy = mae(param_sol_final[1,:], noisy_data[1,:])
    
    println("\nParameter Estimation vs Noisy Data:")
    println("RMSE: $(round(param_rmse_noisy, digits=5))")
    println("R²: $(round(param_r2_noisy, digits=5))")
    println("MAE: $(round(param_mae_noisy, digits=5))")
else
    println("\nParameter Estimation metrics unavailable due to invalid solution")
end

# Create a metrics table for visualization
metrics_table = plot(
    title="Comparison of Model Performance",
    showaxis=false,
    grid=false,
    framestyle=:none
)

# Only add table if both models produced valid solutions
if !any(isinf, nn_sol_final) && !any(isinf, param_sol_final)
    table_data = [
        "Model" "RMSE (True)" "RMSE (Noisy)" "R² (True)" "MAE (True)";
        "Neural ODE" round(neural_rmse, digits=5) round(neural_rmse_noisy, digits=5) round(neural_r2, digits=5) round(neural_mae, digits=5);
        "Parameter Estimation" round(param_rmse, digits=5) round(param_rmse_noisy, digits=5) round(param_r2, digits=5) round(param_mae, digits=5)
    ]
    
    # Save metrics to CSV for report
    open("metrics_results.csv", "w") do io
        for i in 1:size(table_data, 1)
            for j in 1:size(table_data, 2)
                print(io, table_data[i, j])
                if j < size(table_data, 2)
                    print(io, ",")
                end
            end
            println(io)
        end 
    end
    println("\nMetrics saved to metrics_results.csv")
end

println("\nScript finished.") # Neural Ordinary Differential Equations for Langmuir Adsorption Kinetics
