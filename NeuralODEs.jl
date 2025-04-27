# Introduction to Neural ODEs
# This example demonstrates a minimal working Neural ODE model 

using DifferentialEquations
using Plots
using Flux
using DiffEqFlux
using Random

# Set a random seed for reproducibility
Random.seed!(123)

# Define the initial condition and time span
u0 = Float32[2.0; 0.0]  # Initial state
tspan = (0.0f0, 1.5f0)  # Time span from 0 to 1.5
datasize = 30
tsteps = range(tspan[1], tspan[2], length=datasize)  # Time points for data collection

# Define the true ODE function to generate synthetic data
function true_ode_func!(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)' * true_A)'
end

# Create an ODE problem
prob_true = ODEProblem(true_ode_func!, u0, tspan)

# Solve the ODE problem to generate the "ground truth" data
sol_true = solve(prob_true, Tsit5(), saveat=tsteps)
ode_data = Array(sol_true)

# Plot the ground truth data
plt = plot(tsteps, ode_data[1,:], label="True x₁(t)", color=:blue, lw=2)
plot!(plt, tsteps, ode_data[2,:], label="True x₂(t)", color=:red, lw=2)
title!("True Dynamics")
xlabel!("Time")
ylabel!("State")
display(plt)

# Define a simple neural network for the ODE function
nn = Flux.Chain(
    Flux.Dense(2, 16, tanh),
    Flux.Dense(16, 2)
)

# Get the initial parameters of the neural network using destructure
p, re = Flux.destructure(nn)

# Function to reconstruct the neural network from flattened parameters
function dudt_nn(u, p, t)
    nn_recon = re(p)
    return nn_recon(u)
end

# Define the Neural ODE
neuralode = NeuralODE(dudt_nn, tspan, Tsit5(), saveat=tsteps)

# Function to predict using the Neural ODE
function predict(p)
    pred = Array(neuralode(u0, p)[1])
    return pred
end

# Define the loss function
function loss(p)
    pred = predict(p)
    loss = sum(abs2, ode_data .- pred)
    return loss
end

# Keep track of losses
training_losses = []

# Callback function to monitor and visualize training
function cb(p)
    l = loss(p)
    push!(training_losses, l)
    println("Current loss: $l")
    
    # Plot results every 10 iterations
    if length(training_losses) % 10 == 0
        pred = predict(p)
        plt = plot(tsteps, ode_data[1,:], label="True x₁(t)", color=:blue, lw=2)
        plot!(plt, tsteps, ode_data[2,:], label="True x₂(t)", color=:red, lw=2)
        plot!(plt, tsteps, pred[1,:], label="Pred x₁(t)", color=:blue, ls=:dash, lw=2)
        plot!(plt, tsteps, pred[2,:], label="Pred x₂(t)", color=:red, ls=:dash, lw=2)
        title!("Neural ODE Training Progress - Loss: $(round(l, digits=4))")
        display(plt)
    end
    return false
end

# Display initial loss
println("Initial loss: ", loss(p))

# Create an optimizer
opt = ADAM(0.05)

# Training loop
epochs = 200
for i in 1:epochs
    gs = Flux.gradient(loss, p)[1]  # Get gradients
    Flux.Optimise.update!(opt, p, gs)  # Update parameters
    cb(p)  # Call callback function
end

# Plot final results
final_pred = predict(p)
plt_final = plot(tsteps, ode_data[1,:], label="True x₁(t)", color=:blue, lw=2)
plot!(plt_final, tsteps, ode_data[2,:], label="True x₂(t)", color=:red, lw=2)
plot!(plt_final, tsteps, final_pred[1,:], label="Neural ODE x₁(t)", color=:blue, ls=:dash, lw=2)
plot!(plt_final, tsteps, final_pred[2,:], label="Neural ODE x₂(t)", color=:red, ls=:dash, lw=2)
title!("Neural ODE vs True ODE")
xlabel!("Time")
ylabel!("State")
savefig(plt_final, "neural_ode_final_comparison.png")
display(plt_final)

# Plot the loss over training
plt_loss = plot(1:length(training_losses), training_losses, label="Training Loss", lw=2, 
                yscale=:log10, xlabel="Iteration", ylabel="Loss", title="Training Progress")
savefig(plt_loss, "neural_ode_training_loss.png")
display(plt_loss)