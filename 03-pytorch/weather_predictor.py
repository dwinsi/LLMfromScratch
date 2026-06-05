"""
Project 3: Rebuilding in PyTorch with a train/validation split.

Rebuilds the same umbrella network from Project 2 using PyTorch.
Adds a validation set to begin surfacing the concept of overfitting.

Architecture: 3 inputs -> 4 hidden neurons -> 1 output
Framework:    PyTorch (WeatherPredictor class, autograd, SGD optimiser)
New concept:  Train/validation split, watching both loss curves
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ---- Training data (same as Projects 1 and 2) ----
training_inputs = torch.tensor([
    [0.9, 0.7, 0.3],   # cloudy, humid, light wind -> yes
    [0.1, 0.2, 0.1],   # clear, dry, calm -> no
    [0.8, 0.6, 0.5],   # cloudy, humid, windy -> yes
    [0.2, 0.3, 0.2],   # mostly clear, low humidity -> no
    [0.7, 0.8, 0.4],   # cloudy, very humid -> yes
    [0.1, 0.1, 0.9],   # clear but windy -> no
], dtype=torch.float32)

training_labels = torch.tensor(
    [[1], [0], [1], [0], [1], [0]],
    dtype=torch.float32
)

# ---- Validation data (new examples, never seen during training) ----
validation_inputs = torch.tensor([
    [0.85, 0.75, 0.20],   # cloudy and humid -> yes
    [0.15, 0.25, 0.15],   # clear and dry -> no
    [0.60, 0.50, 0.60],   # moderately cloudy and windy -> yes
], dtype=torch.float32)

validation_labels = torch.tensor(
    [[1], [0], [1]],
    dtype=torch.float32
)


# ---- Network definition ----
class WeatherPredictor(nn.Module):

    def __init__(self):
        super(WeatherPredictor, self).__init__()
        self.hidden_layer = nn.Linear(3, 4)
        self.output_layer = nn.Linear(4, 1)
        self.activation   = nn.Sigmoid()

    def forward(self, network_input):
        hidden_output  = self.activation(self.hidden_layer(network_input))
        network_output = self.activation(self.output_layer(hidden_output))
        return network_output


network       = WeatherPredictor()
loss_function = nn.MSELoss()
optimiser     = optim.SGD(network.parameters(), lr=0.5)

training_loss_history   = []
validation_loss_history = []
epochs = 5000

for epoch in range(epochs):

    # ---- Training step ----
    network.train()
    optimiser.zero_grad()
    predicted_training_output = network(training_inputs)
    training_loss             = loss_function(predicted_training_output, training_labels)
    training_loss.backward()
    optimiser.step()
    training_loss_history.append(training_loss.item())

    # ---- Validation step (no gradient updates, no learning) ----
    network.eval()
    with torch.no_grad():
        predicted_validation_output = network(validation_inputs)
        validation_loss             = loss_function(predicted_validation_output, validation_labels)
        validation_loss_history.append(validation_loss.item())

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d}  "
              f"training loss: {training_loss.item():.4f}  "
              f"validation loss: {validation_loss.item():.4f}")


# ---- Final predictions on training data ----
print()
print("Training data predictions:")
network.eval()
with torch.no_grad():
    final_training_predictions = network(training_inputs)
    for i, (predicted, actual) in enumerate(zip(final_training_predictions, training_labels)):
        decision = "bring umbrella" if predicted.item() > 0.5 else "leave it"
        correct  = "correct" if round(predicted.item()) == actual.item() else "wrong"
        print(f"  Sample {i+1}: predicted {predicted.item():.4f} -> {decision} ({correct})")

# ---- Final predictions on validation data ----
print()
print("Validation data predictions (unseen examples):")
with torch.no_grad():
    final_validation_predictions = network(validation_inputs)
    for i, (predicted, actual) in enumerate(zip(final_validation_predictions, validation_labels)):
        decision = "bring umbrella" if predicted.item() > 0.5 else "leave it"
        correct  = "correct" if round(predicted.item()) == actual.item() else "wrong"
        print(f"  Sample {i+1}: predicted {predicted.item():.4f} -> {decision} ({correct})")


# ---- Plot training vs validation loss ----
plt.figure(figsize=(10, 5))
plt.plot(training_loss_history,   color='steelblue', linewidth=1.5, label='Training loss')
plt.plot(validation_loss_history, color='tomato',    linewidth=1.5, label='Validation loss', linestyle='--')
plt.title('Training vs Validation Loss', fontsize=13)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Loss (Mean Squared Error)', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.show()

print("\nLoss curve saved to loss_curve.png")