import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from abc import ABC


class NeuralModel(ABC):
    """
    Abstract base class for model
    """

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.training_history = None
        self.parameters = {}

    def get_model_values(self) -> tuple:
        """Get values of a trained model to be able to save it"""

        return self.parameters

    def restore_model(self, parameters) -> None:
        """Restore saved model values"""

        self.parameters = parameters

    def update(self, gradients, learning_rate) -> None:
        """Adjust parameters with gradients to improve accuracy of the model"""
        for c in range(1, self.network_length + 1):
            self.parameters["W{}".format(c)] = (
                self.parameters["W{}".format(c)]
                - learning_rate * gradients["dW{}".format(c)]
            )
            self.parameters["b{}".format(c)] = (
                self.parameters["b{}".format(c)]
                - learning_rate * gradients["db{}".format(c)]
            )

    def log_loss(self, A, y) -> float:
        """Calculate model loss during training iterations"""

        epsilon = 1e-15
        return (
            1
            / len(y)
            * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))
        )

    def forward_propagation(self, X) -> tuple:
        """Calculate activation for each neurons"""

        activations = {"A0": X}

        for c in range(1, self.network_length + 1):
            Z = (
                self.parameters["W" + str(c)].dot(activations["A" + str(c - 1)])
                + self.parameters["b" + str(c)]
            )
            activations["A{}".format(c)] = 1 / (1 + np.exp(-Z))

        return activations

    def backward_propagation(self, y, activations) -> tuple:
        """Calculate gradients during training to adjust model parameters"""

        m = y.shape[1]

        dZ = activations["A{}".format(self.network_length)] - y
        gradients = {}

        for c in reversed(range(1, self.network_length + 1)):
            gradients["dW{}".format(str(c))] = (
                1 / m * np.dot(dZ, activations["A{}".format(c - 1)].T)
            )
            gradients["db{}".format(str(c))] = 1 / m * np.sum(dZ, axis=1, keepdims=True)

            if c > 1:
                dZ = (
                    np.dot(self.parameters["W" + str(c)].T, dZ)
                    * activations["A" + str(c - 1)]
                    * (1 - activations["A" + str(c - 1)])
                )

        return gradients

    def predict(self, X) -> bool:
        """Predict classes of a dataset X"""

        activations = self.forward_propagation(X)
        Af = activations["A" + str(self.network_length)]

        return Af >= 0.5

    def show_training_results(self) -> None:
        """Display loss and accuracy graphics of the training"""

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history[:, 0], label="Train Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history[:, 1], label="Train Accuracy")
        plt.legend()
        plt.show()

    # override
    def start_training(self, learning_rate=0.1, iteration=200) -> None:
        pass


class MultiLayerNeuralModel(NeuralModel):
    """
    Multiple layers with multiple neurons
    """

    def __init__(self, X_train, y_train, hidden_layers=(32, 32, 32)):
        super().__init__(X_train, y_train)

        dimensions = list(hidden_layers)
        dimensions.insert(0, X_train.shape[0])
        dimensions.append(y_train.shape[0])

        for c in range(1, len(dimensions)):
            self.parameters["W{}".format(c)] = np.random.randn(
                dimensions[c], dimensions[c - 1]
            )
            self.parameters["b{}".format(c)] = np.random.randn(dimensions[c], 1)

        self.network_length = len(self.parameters) // 2

    def start_training(self, learning_rate=0.1, iteration=200) -> None:
        self.training_history = np.zeros((int(iteration), 2))

        for i in tqdm(range(iteration)):
            activations = self.forward_propagation(self.X_train)
            gradients = self.backward_propagation(self.y_train, activations)
            self.update(gradients, learning_rate)
            Af = activations["A{}".format(self.network_length)]

            if i % 10 == 0:
                self.training_history[i, 0] = self.log_loss(
                    self.y_train.flatten(), Af.flatten()
                )
                y_pred = self.predict(self.X_train)
                self.training_history[i, 1] = accuracy_score(
                    self.y_train.flatten(), y_pred.flatten()
                )


class SingleLayerNeuralModel(NeuralModel):
    """
    One layer with one neuron for learning purpose
    """

    def __init__(self, X_train, y_train):
        super().__init__(X_train, y_train)

        self.parameters["W"] = np.random.randn(X_train.shape[1], 1)
        self.parameters["b"] = np.random.randn(1)

    # neuron activation
    def neuron(self, X) -> list:
        Z = X.dot(self.parameters["W"]) + self.parameters["b"]
        A = 1 / (1 + np.exp(-Z))

        return A

    def gradients(self, A, X, y) -> tuple:
        dW = 1 / len(y) * np.dot(X.T, A - y)
        db = 1 / len(y) * np.sum(A - y)

        return (dW, db)

    def update(self, dW, db, learning_rate) -> None:
        self.parameters["W"] -= learning_rate * dW
        self.parameters["b"] -= learning_rate * db

    def predict(self, X) -> bool:
        A = self.neuron(X)

        return A >= 0.5

    def start_training(self, learning_rate=0.1, iteration=200) -> None:
        self.training_history = np.zeros((int(iteration), 2))

        for i in tqdm(range(iteration)):
            A_train = self.neuron(self.X_train)

            if i % 10 == 0:
                self.training_history[i, 0] = self.log_loss(A_train, self.y_train)
                train_y_pred = self.predict(self.X_train)
                self.training_history[i, 1] = accuracy_score(self.y_train, train_y_pred)

            dW, db = self.gradients(A_train, self.X_train, self.y_train)
            self.update(dW, db, learning_rate)
