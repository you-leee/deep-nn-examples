import numpy as np
import matplotlib.pyplot as plt
from NeuralNetworks.regression_funcs.linear_regression import linear_model
from commons.plot_utils.plot_cost import plot_cost


if __name__ == '__main__':
    X = np.array(range(1, 101)) / 51
    X = np.reshape(X, (1, X.shape[0]))
    Y = X * 2 + 4 + (np.random.randn(100) / 2)

    iters = 30000
    lr = 0.00025
    slr_model = linear_model(X, Y, print_cost=True, num_iterations=iters, learning_rate=lr)
    plot_cost(slr_model['costs'], lr)
    print("Weights after {} iterations with learning rate {}: W: {}, b: {}".format(iters, lr, slr_model['w'],
                                                                                   slr_model['b']))

    plt.title('Original vs predicted. Iterations: {}'.format(iters))
    plt.plot(np.squeeze(X), np.squeeze(Y), 'bo', np.squeeze(X), np.squeeze(slr_model['Y_prediction_train']), 'r--')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.show()
