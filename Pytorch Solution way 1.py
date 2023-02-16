# Import pytorch and other libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
a = 0.1 # given
b = 0.2 # given
Pe = 100 # given
Bi = 10 # given
epsilon = 0.01 # given
x0 = 0.5 # given
xi_c = 1 # given

# Define the neural network model
class PDE_Net(nn.Module):
    def __init__(self):
        super(PDE_Net, self).__init__()
        # Define the input, hidden and output layers
        self.input_layer = nn.Linear(2, 10) # input dimension is 2 (theta, Z)
        self.hidden_layer = nn.Linear(10, 10) # hidden dimension is 10
        self.output_layer = nn.Linear(10, 2) # output dimension is 2 (x, xi_c)
        # Define the activation function
        self.activation = nn.Tanh()
    
    def forward(self, theta, Z):
        # Concatenate the inputs
        inputs = torch.cat([theta, Z], 1)
        # Pass through the input layer
        x = self.input_layer(inputs)
        # Pass through the activation function
        x = self.activation(x)
        # Pass through the hidden layer
        x = self.hidden_layer(x)
        # Pass through the activation function
        x = self.activation(x)
        # Pass through the output layer
        x = self.output_layer(x)
        # Split the outputs
        x, xi_c = torch.split(x, 1, 1)
        # Return the outputs
        return x, xi_c

# Create an instance of the model
model = PDE_Net()

# Define the loss function
def loss_fn(x, xi_c, theta, Z):
    # Calculate the partial derivatives using automatic differentiation
    x_theta = torch.autograd.grad(x, theta, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    x_Z = torch.autograd.grad(x, Z, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    x_ZZ = torch.autograd.grad(x_Z, Z, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    xi_c_theta = torch.autograd.grad(xi_c, theta, grad_outputs=torch.ones_like(xi_c), create_graph=True)[0]
    # Calculate the residuals of the pde equations
    residual1 = x_theta + a * x_Z - a / Pe * x_ZZ - (1 - epsilon) / epsilon * 3 * Bi * (x - 1) / (1 - Bi * (1 - 1 / xi_c))
    residual2 = xi_c_theta - b * Bi * (x - 1) / (xi_c**2 * (1 - Bi * (1 - 1 / xi_c)))
    # Calculate the boundary and initial conditions
    bc1 = x - 1 / Pe * x_Z # at Z = 0
    bc2 = x_Z # at Z = 1
    bc3 = x # at theta = 0
    bc4 = x - x0 # at theta = 0
    bc5 = xi_c - 1 # at theta = 0
    # Calculate the total loss as the mean squared error of the residuals and the boundary conditions
    loss = torch.mean(residual1**2) + torch.mean(residual2**2) + torch.mean(bc1**2) + torch.mean(bc2**2) + torch.mean(bc3**2) + torch.mean(bc4**2) + torch.mean(bc5**2)
    # Return the loss
    return loss

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Define the number of epochs
epochs = 1000

# Define the number of samples
n_samples = 100

# Generate random samples of theta and Z
theta = torch.rand(n_samples, 1) * 2 * np.pi # theta in [0, 2*pi]
Z = torch.rand(n_samples, 1) # Z in [0, 1]

# Train the model
for epoch in range(epochs):
    # Zero the gradients
    optimizer.zero_grad()
    # Forward pass

    x, xi_c = model(theta, Z)
    # Calculate the loss
    loss = loss_fn(x, xi_c, theta, Z)
    # Backward pass
    loss.backward()
    # Update the parameters
    optimizer.step()
    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss {loss.item()}")

# Plot the results
theta = torch.linspace(0, 2 * np.pi, 100).reshape(-1, 1) # theta in [0, 2*pi]
Z = torch.linspace(0, 1, 100).reshape(-1, 1) # Z in [0, 1]
x, xi_c = model(theta, Z)
plt.plot(theta.detach().numpy(), x.detach().numpy(), label="x")
plt.plot(theta.detach().numpy(), xi_c.detach().numpy(), label="xi_c")
plt.xlabel("theta")
plt.ylabel("x, xi_c")
plt.legend()
plt.show()