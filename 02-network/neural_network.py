"""
Project 2: A neural network that learns.

Extends Project 1 from a single neuron with hand-picked weights
to a small network that discovers its own weights from labelled data.

Architecture: 3 inputs -> 4 hidden neurons -> 1 output
Training:     Mean squared error loss + backpropagation
Dataset:      6 umbrella weather examples

This script runs two training experiments side by side to show
how adjusting the number of epochs affects the final result.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def sigmoid(x):
    """Squash any number into a value between 0 and 1."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivative of sigmoid with respect to its input.
    If output = sigmoid(x), then derivative = output * (1 - output).
    Used during backpropagation to compute gradients.
    """
    sigmoid_output = sigmoid(x)
    return sigmoid_output * (1 - sigmoid_output)


def train_network(training_inputs, training_labels, epochs, learning_rate=0.5):
    """
    Train a 3 -> 4 -> 1 network for a given number of epochs.
    Returns the final predicted outputs and the full loss history.
    """
    np.random.seed(42)

    weights_input_to_hidden  = np.random.randn(3, 4) * 0.5
    bias_hidden              = np.zeros((1, 4))
    weights_hidden_to_output = np.random.randn(4, 1) * 0.5
    bias_output              = np.zeros((1, 1))

    loss_history = []

    for epoch in range(epochs):

        # ---- Forward pass ----
        hidden_layer_input  = np.dot(training_inputs, weights_input_to_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input  = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
        predicted_output    = sigmoid(output_layer_input)

        # ---- Loss: mean squared error ----
        loss = np.mean((training_labels - predicted_output) ** 2)
        loss_history.append(loss)

        # ---- Backward pass ----

        # Output layer gradients
        gradient_loss_wrt_predicted         = -2 * (training_labels - predicted_output) / len(training_inputs)
        gradient_predicted_wrt_output_input = sigmoid_derivative(output_layer_input)
        gradient_output_delta               = gradient_loss_wrt_predicted * gradient_predicted_wrt_output_input

        gradient_weights_hidden_to_output = np.dot(hidden_layer_output.T, gradient_output_delta)
        gradient_bias_output              = np.sum(gradient_output_delta, axis=0, keepdims=True)

        # Hidden layer gradients
        gradient_loss_wrt_hidden_output  = np.dot(gradient_output_delta, weights_hidden_to_output.T)
        gradient_hidden_wrt_hidden_input = sigmoid_derivative(hidden_layer_input)
        gradient_hidden_delta            = gradient_loss_wrt_hidden_output * gradient_hidden_wrt_hidden_input

        gradient_weights_input_to_hidden = np.dot(training_inputs.T, gradient_hidden_delta)
        gradient_bias_hidden             = np.sum(gradient_hidden_delta, axis=0, keepdims=True)

        # ---- Update weights ----
        weights_input_to_hidden  -= learning_rate * gradient_weights_input_to_hidden
        bias_hidden              -= learning_rate * gradient_bias_hidden
        weights_hidden_to_output -= learning_rate * gradient_weights_hidden_to_output
        bias_output              -= learning_rate * gradient_bias_output

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:5d}  loss: {loss:.4f}")

    return predicted_output, loss_history


# ---- Training data ----
training_inputs = np.array([
    [0.9, 0.7, 0.3],   # cloudy, humid, light wind -> yes
    [0.1, 0.2, 0.1],   # clear, dry, calm -> no
    [0.8, 0.6, 0.5],   # cloudy, humid, windy -> yes
    [0.2, 0.3, 0.2],   # mostly clear, low humidity -> no
    [0.7, 0.8, 0.4],   # cloudy, very humid -> yes
    [0.1, 0.1, 0.9],   # clear but windy -> no
])

training_labels = np.array([[1], [0], [1], [0], [1], [0]])


# ---- Experiment 1: 800 epochs ----
print("=" * 45)
print("Experiment 1: Training for 800 epochs")
print("=" * 45)
predicted_800, loss_history_800 = train_network(training_inputs, training_labels, epochs=800)

print()
print("Predictions after 800 epochs:")
for i, (predicted, actual) in enumerate(zip(predicted_800, training_labels)):
    decision = "bring umbrella" if predicted[0] > 0.5 else "leave it"
    correct  = "correct" if round(predicted[0]) == actual[0] else "wrong"
    print(f"  Sample {i+1}: predicted {predicted[0]:.4f} -> {decision} ({correct})")

print(f"\n  Final loss: {loss_history_800[-1]:.4f}")


# ---- Experiment 2: 5000 epochs ----
print()
print("=" * 45)
print("Experiment 2: Training for 5000 epochs")
print("=" * 45)
predicted_5000, loss_history_5000 = train_network(training_inputs, training_labels, epochs=5000)

print()
print("Predictions after 5000 epochs:")
for i, (predicted, actual) in enumerate(zip(predicted_5000, training_labels)):
    decision = "bring umbrella" if predicted[0] > 0.5 else "leave it"
    correct  = "correct" if round(predicted[0]) == actual[0] else "wrong"
    print(f"  Sample {i+1}: predicted {predicted[0]:.4f} -> {decision} ({correct})")

print(f"\n  Final loss: {loss_history_5000[-1]:.4f}")


# ---- Comparison summary ----
print()
print("=" * 45)
print("Comparison")
print("=" * 45)
print(f"  800 epochs  -> final loss: {loss_history_800[-1]:.4f}")
print(f"  5000 epochs -> final loss: {loss_history_5000[-1]:.4f}")
print(f"  Extra 4200 epochs reduced loss by: {loss_history_800[-1] - loss_history_5000[-1]:.4f}")


# ---- Plot both loss curves side by side ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: 800 epochs
axes[0].plot(loss_history_800, color='steelblue', linewidth=1.5, label='Training loss')
axes[0].set_title('Experiment 1: 800 Epochs', fontsize=13)
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss (Mean Squared Error)', fontsize=11)
axes[0].annotate(f'Start: {loss_history_800[0]:.4f}', xy=(0, loss_history_800[0]),
                 xytext=(50, loss_history_800[0] - 0.02), fontsize=9, color='dimgray')
axes[0].annotate(f'End: {loss_history_800[-1]:.4f}', xy=(799, loss_history_800[-1]),
                 xytext=(550, loss_history_800[-1] + 0.03), fontsize=9, color='dimgray')
axes[0].legend(fontsize=10)

# Right plot: 5000 epochs with 800 epoch marker
axes[1].plot(loss_history_5000, color='steelblue', linewidth=1.5, label='Training loss')
axes[1].axvline(x=800, color='tomato', linestyle='--', linewidth=1, label='Epoch 800')
axes[1].axvline(x=600, color='orange', linestyle='--', linewidth=1, label='Epoch 600 (97.7% learned)')
axes[1].set_title('Experiment 2: 5000 Epochs', fontsize=13)
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Loss (Mean Squared Error)', fontsize=11)
axes[1].annotate(f'Start: {loss_history_5000[0]:.4f}', xy=(0, loss_history_5000[0]),
                 xytext=(100, loss_history_5000[0] - 0.02), fontsize=9, color='dimgray')
axes[1].annotate(f'End: {loss_history_5000[-1]:.4f}', xy=(4999, loss_history_5000[-1]),
                 xytext=(3500, loss_history_5000[-1] + 0.02), fontsize=9, color='dimgray')
axes[1].legend(fontsize=10)

plt.suptitle('Training Comparison: 800 vs 5000 Epochs', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nLoss curve saved to loss_curve.png")