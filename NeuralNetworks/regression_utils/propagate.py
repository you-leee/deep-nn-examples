from . import np
from ..activations.sigmoid import sigmoid


def logistic_propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the logistic propagation

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]

    A, _ = sigmoid(linear_forward_propagate(w, b, X))  # 1 x m   # compute activation
    cost = -(np.dot(Y, np.log(A).T) + np.dot((1 - Y), np.log(1 - A).T)) / m  # 1 x 1     # compute cost

    grads = linear_backward_propagate(X, A, Y, m)

    return grads, cost


def linear_propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the linear propagation

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    gradcheck -- whtere to do gradient checking
    epsilon -- gradient checking parameter

    Return:
    cost -- cost for linear regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]

    A = linear_forward_propagate(w, b, X)  # 1 x m    # compute activation

    # compute cost
    cost = np.sum(np.square(A - Y)) / m  # 1 x 1
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = linear_backward_propagate(X, A, Y, m)

    return grads, cost


def linear_forward_propagate(w, b, X):
    return np.dot(w.T, X) + b  # 1 x m    # compute activation


def linear_backward_propagate(X, A, Y, m):
    dz = A - Y  # 1 x m
    dw = np.dot(X, dz.T) / m  # nx x 1
    db = np.sum(dz) / m  # 1 x 1

    assert (db.dtype == float)

    grads = {"dw": dw,
             "db": db}

    return grads
