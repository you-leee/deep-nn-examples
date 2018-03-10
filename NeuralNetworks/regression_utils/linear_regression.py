from .predict import linear_predict
from .propagate import linear_propagate
from .optimize import optimize
from . import np


def linear_model(X_train, Y_train, X_test=None, Y_test=None, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (n_x, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (n_x, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """
    # initialize parameters with zeros
    w = np.zeros((X_train.shape[0], 1))  # n_x x 1
    b = 0

    # Gradient descent
    params, costs = optimize(X_train, Y_train, w, b, num_iterations, learning_rate, linear_propagate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = params["w"]
    b = params["b"]

    # Return values
    d = {"costs": costs,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    # Predict test/train set examples and print
    Y_prediction_train = linear_predict(w, b, X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    d["Y_prediction_train"] = Y_prediction_train

    if X_test and Y_test:
        Y_prediction_test = linear_predict(w, b, X_test)
        d["Y_prediction_test"] = Y_prediction_test
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    return d
