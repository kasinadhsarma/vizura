# Import necessary packages
using DifferentialEquations
using Plots
using OrdinaryDiffEq
using Flux
using ComponentArrays
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using DiffEqFlux
using Zygote
using Random
using Statistics
using BSON: @save, @load

# Set random seed for reproducibility
Random.seed!(123)

"""
Module for Neural ODE modeling of Langmuir adsorption kinetics
Improved version with better organization, consistent types, and more robust error handling
"""
module LangmuirModels

export generate_data, train_neural_ode, train_parametric_model, evaluate_models, rmse, r_squared, mae

using DifferentialEquations, Flux, DiffEqFlux, Optimization
using OptimizationOptimJL, OptimizationOptimisers, Random, Statistics
using Plots, Zygote

# --- Error Metrics Functions ---
"""
Root Mean Square Error (RMSE)
"""
function rmse(predictions, targets)
    sqrt(sum((predictions .- targets).^2) / length(predictions))
end

"""
Coefficient of determination (R²)
"""
function r_squared(predictions, targets)
    ss_res = sum((targets .- predictions).^2)
    ss_tot = sum((targets .- mean(targets)).^2)
    return 1 - ss_res / ss_tot
end

"""
Mean Absolute Error (MAE)
"""
function mae(predictions, targets)
    mean(abs.(predictions .- targets))
end

"""
Generate synthetic data from Langmuir model with known parameters
"""
function generate_data(;
    ka_true=0.5,           # adsorption rate constant
    kd_true=0.1,           # desorption rate constant
    P=1.0,                 # pressure (constant)
    theta0=[0.0],          # initial condition
    tspan=(0.0, 20.0),     # time span
    noise_level=0.01,      # noise level for synthetic data
    saveat=0.5,            # save at these time points
    use_float32=false      # whether to use Float32 (true) or Float64 (false)
)
    # Convert types if using Float32
    if use_float32
        theta0 = Float32.(theta0)
        tspan = Float32.(tspan)
        ka_true = Float32(ka_true)
        kd_true = Float32(kd_true)
        P = Float32(P)
        noise_level = Float32(noise_level)
        saveat = Float32(saveat)
    end
    
    # Define the Langmuir ODE function
    function langmuir_ode!(dθ, θ, p, t)
        dθ[1] = ka_true * P * (1 - θ[1]) - kd_true * θ[1]
    end
    
    # Create and solve the ODE problem
    prob = ODEProblem(langmuir_ode!, theta0, tspan)
    sol = solve(prob, Tsit5(), saveat=saveat)
    
    # Extract solution and time points
    t_points = sol.t
    true_solution = Array(sol)
    
    # Add noise
    if use_float32
        noisy_data = Array{Float32}(true_solution) + noise_level * randn(Float32, size(true_solution))
    else
        noisy_data = Array(true_solution) + noise_level * randn(size(true_solution))
    end
    
    return t_points, true_solution, noisy_data, P, ka_true, kd_true
end

"""
Train a Neural ODE model on the provided data
"""
function train_neural_ode(t_points, noisy_data, theta0, tspan; 
                          hidden_dims=[10, 10],       # hidden layer dimensions
                          activation=tanh,            # activation function
                          learning_rate=0.01,         # learning rate for optimizer
                          max_iterations=300,         # maximum training iterations
                          use_float32=false,          # whether to use Float32
                          early_stop_patience=50      # early stopping patience
                         )
    # Type conversion if using Float32
    num_type = use_float32 ? Float32 : Float64
    
    # Build neural network architecture
    layers = []
    prev_dim = 1  # Input dimension
    
    for dim in hidden_dims
        push!(layers, Flux.Dense(prev_dim => dim, activation; 
                               init=Flux.glorot_uniform(gain=num_type(1.0)), bias=true))
        prev_dim = dim
    end
    
    # Output layer
    push!(layers, Flux.Dense(prev_dim => 1; 
                           init=Flux.glorot_uniform(gain=num_type(1.0)), bias=true))
    
    nn_architecture = Flux.Chain(layers...)
    
    # Get initial parameters
    p_nn_init, re = Flux.destructure(nn_architecture)
    
    # Convert to correct number type
    p_nn_init = num_type.(p_nn_init)
    
    # Neural ODE dynamics function
    function nn_dynamics!(dθ, θ, p, t)
        nn_local = re(p)
        nn_output = nn_local(θ)
        dθ .= nn_output
    end
    
    # Create the Neural ODE problem
    nn_prob = ODEProblem{true}(nn_dynamics!, theta0, tspan, p_nn_init)
    
    # Prediction function
    function predict_neural(p)
        _prob = remake(nn_prob, p=p)
        
        # Use try-catch for more robust error handling
        try
            sol = solve(_prob, Tsit5(), saveat=t_points, 
                      sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), 
                      abstol=1e-6, reltol=1e-6)
            
            if sol.retcode != ReturnCode.Success
                return fill(num_type(Inf), size(noisy_data))
            end
            
            return Array(sol)
        catch e
            println("Error in Neural ODE solver: ", e)
            return fill(num_type(Inf), size(noisy_data))
        end
    end
    
    # Loss function
    function loss_neural(p, _)
        pred = predict_neural(p)
        if any(isinf, pred) || any(isnan, pred)
            return num_type(Inf)
        end
        sum(abs2, pred .- noisy_data) / length(noisy_data)
    end
    
    # Training progress tracking
    iterations = Ref(0)
    best_loss = Ref(Inf)
    best_params = copy(p_nn_init)
    patience_counter = 0
    
    callback = function (p, l)
        iterations[] += 1
        
        # Track best parameters
        if l < best_loss[]
            best_loss[] = l
            best_params .= p
            patience_counter = 0
        else
            patience_counter += 1
        end
        
        # Early stopping
        if patience_counter >= early_stop_patience
            println("Early stopping at iteration $(iterations[])")
            return true
        end
        
        # Print progress
        if iterations[] % 10 == 0 || iterations[] == 1
            println("Iteration: $(iterations[]), Loss: $l")
            if isnan(l) || isinf(l)
                println("Loss is NaN or Inf, stopping training.")
                return true
            end
        end
        
        return false
    end
    
    # Setup optimization
    optf_neural = OptimizationFunction(loss_neural, Optimization.AutoZygote())
    optprob_neural = OptimizationProblem(optf_neural, p_nn_init, nothing)
    
    # Solve optimization problem
    println("Starting Neural ODE training...")
    result_neural = solve(optprob_neural, ADAM(num_type(learning_rate)), 
                        callback=callback, maxiters=max_iterations)
    println("Neural ODE training finished.")
    
    # Use best parameters found during training
    final_params = best_loss[] < Inf ? best_params : result_neural.u
    
    # Final prediction
    nn_sol_final = predict_neural(final_params)
    
    return final_params, nn_sol_final, re
end

"""
Train a parametric model (traditional approach) on the provided data
"""
function train_parametric_model(t_points, noisy_data, theta0, tspan, P;
                              initial_params=[0.3, 0.2],    # initial parameter guess [ka, kd]
                              learning_rate=0.01,           # learning rate for optimizer
                              max_iterations=300,           # maximum training iterations
                              use_float32=false,            # whether to use Float32
                              early_stop_patience=50,       # early stopping patience
                              use_bfgs=false                # whether to use BFGS or ADAM
                             )
    # Type conversion if using Float32
    num_type = use_float32 ? Float32 : Float64
    
    # Convert initial parameters
    p_initial = num_type.(initial_params)
    
    # Define the Langmuir ODE function with parameters
    function langmuir_ode_param!(dθ, θ, p, t)
        ka, kd = p
        P_local = num_type(P)  # Ensure P is the right type
        dθ[1] = ka * P_local * (1 - θ[1]) - kd * θ[1]
    end
    
    # Create ODE problem
    param_prob = ODEProblem{true}(langmuir_ode_param!, theta0, tspan, p_initial)
    
    # Prediction function
    function predict_param(p)
        _prob = remake(param_prob, p=p)
        
        # Use try-catch for more robust error handling
        try
            sol = solve(_prob, Tsit5(), saveat=t_points, abstol=1e-6, reltol=1e-6)
            
            if sol.retcode != ReturnCode.Success
                return fill(num_type(Inf), size(noisy_data))
            end
            
            return Array(sol)
        catch e
            println("Error in parametric model solver: ", e)
            return fill(num_type(Inf), size(noisy_data))
        end
    end
    
    # Loss function
    function loss_param(p, _)
        # Basic parameter constraints (prevent negative values)
        if any(p .< 0)
            return num_type(Inf)
        end
        
        pred = predict_param(p)
        if any(isinf, pred) || any(isnan, pred)
            return num_type(Inf)
        end
        sum(abs2, pred .- noisy_data) / length(noisy_data)
    end
    
    # Training progress tracking
    iterations = Ref(0)
    best_loss = Ref(Inf)
    best_params = copy(p_initial)
    patience_counter = 0
    
    callback = function (p, l)
        iterations[] += 1
        
        # Track best parameters
        if l < best_loss[]
            best_loss[] = l
            best_params .= p
            patience_counter = 0
        else
            patience_counter += 1
        end
        
        # Early stopping
        if patience_counter >= early_stop_patience
            println("Early stopping at iteration $(iterations[])")
            return true
        end
        
        # Print progress
        if iterations[] % 10 == 0 || iterations[] == 1
            println("Iteration: $(iterations[]), Loss: $l")
            if isnan(l) || isinf(l)
                println("Loss is NaN or Inf, stopping training.")
                return true
            end
        end
        
        return false
    end
    
    # Setup optimization
    optf_param = OptimizationFunction(loss_param, Optimization.AutoZygote())
    optprob_param = OptimizationProblem(optf_param, p_initial, nothing)
    
    # Solve optimization problem - choose optimizer based on parameter
    println("Training parametric model...")
    if use_bfgs
        # Switch to Float64 if using BFGS
        if use_float32
            println("Note: Converting to Float64 for BFGS compatibility")
            theta0_64 = Float64.(theta0)
            tspan_64 = Float64.(tspan)
            noisy_data_64 = Float64.(noisy_data)
            t_points_64 = Float64.(t_points)
            p_initial_64 = Float64.(p_initial)
            
            # Redefine problem with Float64
            param_prob_64 = ODEProblem(langmuir_ode_param!, theta0_64, tspan_64, p_initial_64)
            
            # Redefine loss and optimization setup
            # ... (code would go here)
            
            result_param = solve(optprob_param, BFGS(initial_stepnorm=0.01), 
                               callback=callback, maxiters=max_iterations)
        else
            result_param = solve(optprob_param, BFGS(initial_stepnorm=0.01), 
                               callback=callback, maxiters=max_iterations)
        end
    else
        result_param = solve(optprob_param, ADAM(num_type(learning_rate)), 
                           callback=callback, maxiters=max_iterations)
    end
    println("Parameter estimation training finished.")
    
    # Use best parameters found during training
    final_params = best_loss[] < Inf ? best_params : result_param.u
    
    # Final prediction
    param_sol_final = predict_param(final_params)
    
    return final_params, param_sol_final
end

"""
Calculate error metrics and create comparison visualizations
"""
function evaluate_models(t_points, true_solution, noisy_data, nn_sol_final, param_sol_final, 
                       learned_params, ka_true, kd_true; save_plots=true, save_metrics=true)
    
    metrics = Dict()
    
    # Check if both solutions are valid
    neural_valid = !any(isinf, nn_sol_final) && !any(isnan, nn_sol_final)
    param_valid = !any(isinf, param_sol_final) && !any(isnan, param_sol_final)
    
    println("\n=== ERROR METRICS ===")
    
    # Neural ODE metrics
    if neural_valid
        neural_rmse_true = rmse(nn_sol_final[1,:], true_solution[1,:])
        neural_r2_true = r_squared(nn_sol_final[1,:], true_solution[1,:])
        neural_mae_true = mae(nn_sol_final[1,:], true_solution[1,:])
        
        neural_rmse_noisy = rmse(nn_sol_final[1,:], noisy_data[1,:])
        neural_r2_noisy = r_squared(nn_sol_final[1,:], noisy_data[1,:])
        neural_mae_noisy = mae(nn_sol_final[1,:], noisy_data[1,:])
        
        println("\nNeural ODE vs True Solution:")
        println("RMSE: $(round(neural_rmse_true, digits=5))")
        println("R²: $(round(neural_r2_true, digits=5))")
        println("MAE: $(round(neural_mae_true, digits=5))")
        
        println("\nNeural ODE vs Noisy Data:")
        println("RMSE: $(round(neural_rmse_noisy, digits=5))")
        println("R²: $(round(neural_r2_noisy, digits=5))")
        println("MAE: $(round(neural_mae_noisy, digits=5))")
        
        # Store metrics
        metrics["neural_rmse_true"] = neural_rmse_true
        metrics["neural_r2_true"] = neural_r2_true
        metrics["neural_mae_true"] = neural_mae_true
        metrics["neural_rmse_noisy"] = neural_rmse_noisy
        metrics["neural_r2_noisy"] = neural_r2_noisy
        metrics["neural_mae_noisy"] = neural_mae_noisy
    else
        println("\nNeural ODE metrics unavailable due to invalid solution")
    end
    
    # Parameter estimation metrics
    if param_valid
        param_rmse_true = rmse(param_sol_final[1,:], true_solution[1,:])
        param_r2_true = r_squared(param_sol_final[1,:], true_solution[1,:])
        param_mae_true = mae(param_sol_final[1,:], true_solution[1,:])
        
        param_rmse_noisy = rmse(param_sol_final[1,:], noisy_data[1,:])
        param_r2_noisy = r_squared(param_sol_final[1,:], noisy_data[1,:])
        param_mae_noisy = mae(param_sol_final[1,:], noisy_data[1,:])
        
        println("\nParameter Estimation vs True Solution:")
        println("RMSE: $(round(param_rmse_true, digits=5))")
        println("R²: $(round(param_r2_true, digits=5))")
        println("MAE: $(round(param_mae_true, digits=5))")
        
        println("\nParameter Estimation vs Noisy Data:")
        println("RMSE: $(round(param_rmse_noisy, digits=5))")
        println("R²: $(round(param_r2_noisy, digits=5))")
        println("MAE: $(round(param_mae_noisy, digits=5))")
        
        # Store metrics
        metrics["param_rmse_true"] = param_rmse_true
        metrics["param_r2_true"] = param_r2_true
        metrics["param_mae_true"] = param_mae_true
        metrics["param_rmse_noisy"] = param_rmse_noisy
        metrics["param_r2_noisy"] = param_r2_noisy
        metrics["param_mae_noisy"] = param_mae_noisy
        
        # Parameter recovery results
        ka_learned, kd_learned = learned_params
        ka_error = abs(ka_learned - ka_true)/ka_true * 100
        kd_error = abs(kd_learned - kd_true)/kd_true * 100
        
        println("\nParameter Recovery:")
        println("True ka: $ka_true, Learned ka: $ka_learned ($(round(ka_error, digits=2))% error)")
        println("True kd: $kd_true, Learned kd: $kd_learned ($(round(kd_error, digits=2))% error)")
        
        # Store parameter recovery metrics
        metrics["ka_true"] = ka_true
        metrics["kd_true"] = kd_true
        metrics["ka_learned"] = ka_learned
        metrics["kd_learned"] = kd_learned
        metrics["ka_error"] = ka_error
        metrics["kd_error"] = kd_error
    else
        println("\nParameter Estimation metrics unavailable due to invalid solution")
    end
    
    # Plot comparison
    if save_plots
        # Comparison plot
        plt_comparison = plot(t_points, true_solution[1,:], label="True Solution",
                           title="Parameter Learning vs Neural ODE",
                           xlabel="Time", ylabel="Surface Coverage (θ)",
                           legend=:bottomright, lw=2)
        scatter!(plt_comparison, t_points, noisy_data[1,:], label="Noisy Data", alpha=0.5)
        
        if neural_valid
            plot!(plt_comparison, t_points, nn_sol_final[1,:], label="Neural ODE", ls=:dash, lw=2)
        end
        
        if param_valid
            plot!(plt_comparison, t_points, param_sol_final[1,:], label="Learned Parameters", ls=:dot, lw=2)
        end
        
        savefig(plt_comparison, "langmuir_comparison_improved.png")
        println("\nComparison plot saved to langmuir_comparison_improved.png")
    end
    
    # Save metrics to CSV
    if save_metrics && neural_valid && param_valid
        open("metrics_results_improved.csv", "w") do io
            # Headers
            println(io, "Model,RMSE (True),RMSE (Noisy),R² (True),R² (Noisy),MAE (True),MAE (Noisy)")
            # Neural ODE data
            println(io, "Neural ODE,$(round(metrics["neural_rmse_true"], digits=5)),$(round(metrics["neural_rmse_noisy"], digits=5)),$(round(metrics["neural_r2_true"], digits=5)),$(round(metrics["neural_r2_noisy"], digits=5)),$(round(metrics["neural_mae_true"], digits=5)),$(round(metrics["neural_mae_noisy"], digits=5))")
            # Parameter Estimation data
            println(io, "Parameter Estimation,$(round(metrics["param_rmse_true"], digits=5)),$(round(metrics["param_rmse_noisy"], digits=5)),$(round(metrics["param_r2_true"], digits=5)),$(round(metrics["param_r2_noisy"], digits=5)),$(round(metrics["param_mae_true"], digits=5)),$(round(metrics["param_mae_noisy"], digits=5))")
        end
        println("Metrics saved to metrics_results_improved.csv")
        
        # Save parameter recovery results
        open("parameter_recovery.csv", "w") do io
            println(io, "Parameter,True Value,Estimated Value,Error (%)")
            println(io, "ka,$(metrics["ka_true"]),$(metrics["ka_learned"]),$(round(metrics["ka_error"], digits=2))")
            println(io, "kd,$(metrics["kd_true"]),$(metrics["kd_learned"]),$(round(metrics["kd_error"], digits=2))")
        end
        println("Parameter recovery results saved to parameter_recovery.csv")
    end
    
    return metrics
end

end # module LangmuirModels

# --- Main Script to Run the Analysis ---
using .LangmuirModels

function run_langmuir_analysis()
    println("Starting Langmuir Adsorption Neural ODE Analysis")
    
    # Choose whether to use Float32 or Float64
    use_float32 = false  # Set to true for Float32, false for Float64
    num_type = use_float32 ? Float32 : Float64
    
    # Generate synthetic data
    println("Generating synthetic data...")
    t_points, true_solution, noisy_data, P, ka_true, kd_true = generate_data(
        use_float32=use_float32
    )
    
    # Plot the data
    plt = plot(t_points, true_solution[1,:], label="True Solution",
               title="Langmuir Adsorption Model",
               xlabel="Time", ylabel="Surface Coverage (θ)",
               legend=:bottomright, lw=2)
    scatter!(plt, t_points, noisy_data[1,:], label="Noisy Data", alpha=0.5)
    savefig(plt, "langmuir_data.png")
    println("Data plot saved to langmuir_data.png")
    
    # Initial condition and time span
    theta0 = num_type[0.0]
    tspan = num_type.((0.0, 20.0))
    
    # Train Neural ODE
    println("\n--- Neural ODE Training ---")
    # Experiment with different architectures
    hidden_dims = [10, 10]  # Default architecture: two hidden layers with 10 neurons each
    
    neural_params, nn_sol_final, re = train_neural_ode(
        t_points, noisy_data, theta0, tspan;
        hidden_dims=hidden_dims,
        learning_rate=0.01,
        max_iterations=300,
        use_float32=use_float32,
        early_stop_patience=50
    )
    
    # Save the trained neural network
    @save "neural_ode_model.bson" neural_params re
    
    # Train Parametric Model
    println("\n--- Parametric Model Training ---")
    learned_params, param_sol_final = train_parametric_model(
        t_points, noisy_data, theta0, tspan, P;
        initial_params=[0.3, 0.2],
        learning_rate=0.01,
        max_iterations=300,
        use_float32=use_float32,
        early_stop_patience=50,
        use_bfgs=false  # Set to true to try BFGS optimizer
    )
    
    # Save the learned parameters
    @save "parametric_model.bson" learned_params
    
    # Evaluate both models
    metrics = i in 1:size(table_data, 1)(
        t_points, true_solution, noisy_data, 
        nn_sol_final, param_sol_final, 
        learned_params, ka_true, kd_true;
        save_plots=true,
        save_metrics=true
    )
    
    println("\nAnalysis complete!")
    return metrics
end

# Run the analysis
metrics = run_langmuir_analysis()