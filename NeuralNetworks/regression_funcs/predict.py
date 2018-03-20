from . import np
from NeuralNetworks.activations.sigmoid import sigmoid


def logistic_predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (n_x, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A, _ = sigmoid(np.dot(w.T, X) + b)

    Y_prediction = (A > 0.5).astype(int)

    assert (Y_prediction.shape == (1, X.shape[1]))
    return Y_prediction


def linear_predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned linear regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = np.dot(w.T, X) + b

    assert (A.shape == (1, X.shape[1]))
    return A
