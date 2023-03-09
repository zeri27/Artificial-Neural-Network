import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class ANN:
    def __init__(self, num_features, hidden_neurons, target_classes, learning_rate, epoch):
        self.weights1 = None
        self.weights2 = None
        # 10 features in the data
        self.input_layer_size = num_features
        self.hidden_neurons = hidden_neurons
        # 7 target classes
        self.output_layer_size = target_classes
        self.learning_rate = learning_rate
        self.epoch = epoch

    # Softmax activation function
    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x))
        return exps / (np.sum(exps, axis=1, keepdims=True) + 1e-6)

    # Softmax derivative activation function
    def softmax_derivative(self, x):
        softmax_x = self.softmax(x)
        return softmax_x * (1 - softmax_x)

    # Tan H activation function
    @staticmethod
    def tanh(x):
        return np.tanh(x)

    # Tan H derivative activation function
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    # Function to train the model given a set of features and target classes
    def train(self, x_train, y_train, x_valid, y_valid, plot_graph, early_stopping):
        y_train_remodelled = pd.get_dummies(y_train).values

        # Initialize the weights for layer between input and hidden
        weights1 = 2 * np.random.random((self.input_layer_size, self.hidden_neurons)) - 1
        self.weights1 = weights1
        # Initialize the weights for layer between hidden and output
        weights2 = 2 * np.random.random((self.hidden_neurons, self.output_layer_size)) - 1
        self.weights2 = weights2

        best_weights1 = weights1
        best_weights2 = weights2
        best_val_acc = 0.0

        # Keep track of training and validation accuracy across epochs
        train_acc_list = []
        val_acc_list = []

        # Train the neural network
        for i in range(self.epoch):
            for j in range(len(x_train)):
                # Forward propagation
                layer1_output = self.tanh(np.dot(x_train[j:j + 1], weights1))
                layer2_output = self.softmax(np.dot(layer1_output, weights2))

                # Backpropagation
                layer2_error = y_train_remodelled[j:j + 1] - layer2_output
                layer2_delta = layer2_error * self.softmax_derivative(layer2_output)

                layer1_error = layer2_delta.dot(weights2.T)
                layer1_delta = layer1_error * self.tanh_derivative(layer1_output)

                # Update the weights
                weights2 += self.learning_rate * layer1_output.T.dot(layer2_delta)
                weights1 += self.learning_rate * x_train[j:j + 1].T.dot(layer1_delta)

            # Evaluate performance on training set
            train_pred = self.predict(x_train)
            train_acc = np.mean(np.array(y_train) == np.array(train_pred))

            # Evaluate performance on validation set
            val_pred = self.predict(x_valid)
            val_acc = np.mean(np.array(y_valid) == np.array(val_pred))

            # Update best weights if validation accuracy improves
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights1 = weights1
                best_weights2 = weights2

            # Store training and validation accuracy at current epoch
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)

            if early_stopping and best_val_acc >= 0.85:
                self.epoch = i + 1
                break

        # Store the best set of weights
        self.weights1 = best_weights1
        self.weights2 = best_weights2

        if plot_graph:
            self.plotGraph(train_acc_list, val_acc_list)

        return best_val_acc

    # Predict the target classes for new data
    def predict(self, new_data):
        layer1_output = self.tanh(np.dot(new_data, self.weights1))
        layer2_output = self.softmax(np.dot(layer1_output, self.weights2))
        return np.argmax(layer2_output, axis=1) + 1

    # Plot graph for the training and validation accuracies per epoch
    def plotGraph(self, train_acc_list, val_acc_list):
        # Plot the training and validation accuracy across epochs
        plt.title('Training & Validation Accuracy Across Epochs')
        plt.plot(range(1, self.epoch + 1), train_acc_list, label='Training Accuracy')
        plt.plot(range(1, self.epoch + 1), val_acc_list, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()