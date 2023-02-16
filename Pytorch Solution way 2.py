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
eps = 0.01 # given
x0 = 0.5 # given
xi0 = 1 # given

# Define the neural network model
class PDE_Net(nn.Module):
    def __init__(self):
        super(PDE_Net, self).__init__()
        # Define the network architecture
        self.fc1 = nn.Linear(2, 20) # input layer
        self.fc2 = nn.Linear(20, 20) # hidden layer
        self.fc3 = nn.Linear(20, 3) # output layer
        # Define the activation function
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Forward pass
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the model
model = PDE_Net()

# Define the loss function
def loss_fn(x, theta, z):
    # Get the outputs from the model
    x, xi, y = model(torch.cat([theta, z], 1)).split(1, 1)
    # Calculate the derivatives using autograd
    x_theta = torch.autograd.grad(x.sum(), theta, create_graph=True)[0]
    x_z = torch.autograd.grad(x.sum(), z, create_graph=True)[0]
    x_zz = torch.autograd.grad(x_z.sum(), z, create_graph=True)[0]
    xi_theta = torch.autograd.grad(xi.sum(), theta, create_graph=True)[0]
    # Calculate the residuals of the pde equations
    r1 = x_theta + a * x_z - a / Pe * x_zz - (1 - eps) / eps * 3 * Bi * (x - 1) / (1 - Bi * (1 - 1 / xi))
    r2 = xi_theta - b * Bi * (x - 1) / (xi**2 * (1 - Bi * (1 - 1 / xi)))
    r3 = y - xi**3
    # Calculate the boundary and initial conditions
    bc1 = x - 1 / Pe * x_z # at z = 0
    bc2 = x_z # at z = 1
    bc3 = x # at theta = 0
    bc4 = x - x0 # at theta = 0, z = 0
    bc5 = xi - xi0 # at theta = 0
    # Calculate the total loss as the mean squared error of the residuals and the boundary conditions
    loss = torch.mean(r1**2) + torch.mean(r2**2) + torch.mean(r3**2) + torch.mean(bc1**2) + torch.mean(bc2**2) + torch.mean(bc3**2) + torch.mean(bc4**2) + torch.mean(bc5**2)
    return loss

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Define the training loop
def train(n_epochs):
    # Loop over the epochs
    for epoch in range(n_epochs):
        # Generate random samples of theta and z in the domain [0, 1] x [0, 1]
        theta = torch.rand(100, 1)
        z = torch.rand(100, 1)
        # Zero the gradients
        optimizer.zero_grad()
        # Calculate the loss
        loss = loss_fn(x, theta, z)
        # Perform backpropagation
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Print the loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Loss {loss.item():.4f}')

# Train the model for 1000 epochs
train(1000)

# Plot the results
# Generate a grid of theta and z values
theta = np.linspace(0, 1, 100)
z = np.linspace(0, 1, 100)
theta, z = np.meshgrid(theta, z)
# Convert to tensors
theta = torch.tensor(theta, dtype=torch.float32)
z = torch.tensor(z, dtype=torch.float32)
  

# Get the outputs from the model
x, xi, y = model(torch.cat([theta.reshape(-1, 1), z.reshape(-1, 1)], 1)).split(1, 1)
# Reshape to the grid shape
x = x.reshape(100, 100)
xi = xi.reshape(100, 100)
y = y.reshape(100, 100)
# Plot the outputs using matplotlib
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.contourf(theta, z, x, cmap='jet')
plt.xlabel('theta')
plt.ylabel('z')
plt.title('x')
plt.colorbar()
plt.subplot(2, 2, 2)
plt.contourf(theta, z, xi, cmap='jet')
plt.xlabel('theta')
plt.ylabel('z')
plt.title('xi')
plt.colorbar()
plt.subplot(2, 2, 3)
plt.contourf(theta, z, y, cmap='jet')
plt.xlabel('theta')
plt.ylabel('z')
plt.title('y')
plt.colorbar()
plt.tight_layout()
plt.show()