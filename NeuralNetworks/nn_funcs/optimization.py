from . import np


def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    Arguments:
    parameters -- python dictionary containing your parameters.

    Returns:
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        W_shape = parameters["W" + str(l + 1)].shape
        b_shape = parameters["b" + str(l + 1)].shape
        v["dW" + str(l + 1)] = np.zeros((W_shape[0], W_shape[1]))
        v["db" + str(l + 1)] = np.zeros((b_shape[0], b_shape[1]))
        s["dW" + str(l + 1)] = np.zeros((W_shape[0], W_shape[1]))
        s["db" + str(l + 1)] = np.zeros((b_shape[0], b_shape[1]))

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    t -- current epoch
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates
    beta2 -- Exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for l in range(L):
        layer = str(l + 1)

        # Moving average of the gradients.
        v["dW" + layer] = beta1 * v["dW" + layer] + ((1 - beta1) * grads["dW" + layer])
        v["db" + layer] = beta1 * v["db" + layer] + ((1 - beta1) * grads["db" + layer])

        # Compute bias-corrected first moment estimate.
        v_corrected["dW" + layer] = v["dW" + layer] / (1 - np.power(beta1, t))
        v_corrected["db" + layer] = v["db" + layer] / (1 - np.power(beta1, t))

        # Moving average of the squared gradients.
        s["dW" + layer] = beta2 * s["dW" + layer] + ((1 - beta2) * np.square(grads["dW" + layer]))
        s["db" + layer] = beta2 * s["db" + layer] + ((1 - beta2) * np.square(grads["db" + layer]))

        # Compute bias-corrected second raw moment estimate.
        s_corrected["dW" + layer] = s["dW" + layer] / (1 - np.power(beta2, t))
        s_corrected["db" + layer] = s["db" + layer] / (1 - np.power(beta2, t))

        # Update parameters.
        parameters["W" + layer] = parameters["W" + layer] - learning_rate * (v_corrected["dW" + layer] / (np.sqrt(s_corrected["dW" + layer]) + epsilon))
        parameters["b" + layer] = parameters["b" + layer] - learning_rate * (v_corrected["db" + layer] / (np.sqrt(s_corrected["db" + layer]) + epsilon))

    return parameters, v, s


def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.

    Returns:
    v -- python dictionary containing the current velocity.
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}

    # Initialize velocity
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape[0], parameters["W" + str(l + 1)].shape[1]))
        v["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape[0], parameters["b" + str(l + 1)].shape[1]))

    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2  # number of layers in the neural networks

    # Momentum update for each parameter
    for l in range(L):
        # compute velocities
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]
        # update parameters
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

    return parameters, v


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters
