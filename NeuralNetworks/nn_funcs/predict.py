from .propagate import L_model_forward


def predict_binary(parameters, X, pos_thresh=0.5):
    """
    This function is used to predict the results of a  L-layer neural network for binary classification

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    pos_trash -- threshold for positive class

    Returns:
    p -- predictions for the given dataset X - either 0(neg) or 1(pos)
    """

    # Forward propagation
    probas = predict(parameters, X)

    # convert probas to 0/1 predictions
    p = (probas > pos_thresh).astype(int)

    return p

def predict(parameters, X):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    return probas