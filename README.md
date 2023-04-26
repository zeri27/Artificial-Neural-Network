# Artificial Neural Network (ANN) for Multiclass Classification

This is a project that implements an Artificial Neural Network (ANN) for multiclass classification using Python. The ANN is implemented using the Numpy, Pandas, and Matplotlib libraries.

# File Structure

The project has two files:

    `NeuralNetwork.py`: This file contains the implementation of the ANN class that performs the training and prediction tasks.
    `Main.py`: This file contains the main code that loads and preprocesses the dataset and trains and tests the ANN model.

# Dataset

The project uses a dataset provided for CSE 2530 Computational Intelligence, a course in the Bachelor's of Computer Science in Delft University of Technology. The dataset consists of 7000 samples with 10 features each. The target variable has 7 classes.

# ANN Architecture

The ANN has two layers, an input layer, and an output layer. The input layer has the same number of nodes as the number of features in the dataset, and the output layer has the same number of nodes as the number of classes in the target variable. The activation function used for the hidden layer is the hyperbolic tangent function, and the output layer uses the softmax activation function.
Training and Evaluation

The ANN is trained using the backpropagation algorithm, and the weights are updated using the stochastic gradient descent optimizer. The model is trained for a fixed number of epochs, and the training and validation accuracy are recorded for each epoch. The best set of weights is chosen based on the validation accuracy, and the model with the best weights is used for prediction.

The performance of the model is evaluated based on the accuracy on the test dataset.

# How to Run

To run the code, follow the steps below:

1. Clone the repository or download the files.
2. Install the required libraries using the following command: `pip install numpy pandas matplotlib`.
3. Run the `main.py` file using the command python `main.py`.

The program will output the accuracy of the model on the test dataset and show a plot of the training and validation accuracy across epochs.
