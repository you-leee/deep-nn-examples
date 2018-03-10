import matplotlib.pyplot as plt

from NeuralNetworks.dataload_utils.load_planar import load_planar_dataset
from NeuralNetworks.plot_utils.plot_decision_boundary import plot_decision_boundary
from NeuralNetworks.nn_utils.models import L_layer_model
from NeuralNetworks.nn_utils.predict import predict_binary
from NeuralNetworks.regression_utils.logistic_regression import *
from NeuralNetworks.regression_utils.predict import logistic_predict


if __name__ == '__main__':

    np.random.seed(1)  # set a seed so that the results are consistent
    X, Y = load_planar_dataset()

    # Visualize the data:
    plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
    plt.show()

    shape_X = X.shape # n_x x m
    shape_Y = Y.shape # 1 x m
    m = shape_X[1]  # training set size

    # Train the logistic regression classifier
    lr_model = logistic_model(X, Y)
    # Plot the decision boundary for logistic regression
    plot_decision_boundary(lambda x: logistic_predict(lr_model['w'], lr_model['b'], x.T), X, Y)
    plt.title("Logistic Regression")
    plt.show()

    # Print accuracy
    LR_predictions = logistic_predict(lr_model['w'], lr_model['b'], X)
    accuracy = float((np.dot(Y, LR_predictions.T) + np.dot(1 - Y, 1 - LR_predictions.T)) / float(Y.size) * 100)
    print('Accuracy of logistic regression: %d ' % accuracy + '% ' + "(percentage of correctly labelled datapoints)\n")

    # Tuning hidden layer size
    plt.figure(figsize=(16, 32))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20, 50, 100]
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(3, 3, i + 1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = L_layer_model(X, Y, [X.shape[0], n_h, Y.shape[0]], num_epochs=5000, learning_rate=0.1)
        plot_decision_boundary(lambda x: predict_binary(parameters, x.T), X, Y)
        predictions = predict_binary(parameters, X)
        accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
        print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

    plt.show()
