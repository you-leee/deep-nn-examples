from . import np
from .initialization import initialize_parameters_zeros, initialize_parameters_random, initialize_parameters_he
from commons.minibatch_utils.random_mini_batches import random_mini_batches
from .propagate import L_model_forward, compute_cost, L_model_backward
from .optimization import initialize_adam, update_parameters_with_adam, initialize_velocity, \
    update_parameters_with_momentum, update_parameters
from .predict import predict_binary
from commons.plot_utils.plot_cost import plot_cost


def L_layer_model(X_train, Y_train, layers_dims, X_test=None, Y_test=None, num_epochs=1000, mini_batch_size=None,
                  initialization="he", optimizer="gd", beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8,
                  learning_rate=0.001, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X_train -- input train data, of shape (n_x, number of examples)
    Y_train -- true train"label" vector
    layers_dims -- dimensions of the layers (n_x, n_h1, nh2,..., n_y)
    X_test -- input test data, of shape (n_x, number of testcases)
    Y_test -- true test "label" vector
    num_epochs -- number of iterations of the optimization loop
    mini_batch_size -- number of samples in 1 mini-batch
    initialization -- name of initialization method ("he", "random" or "zero")
    optimizer -- name of optimization method ("gd", "momentum" or "adam")
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    seed = 1
    np.random.seed(seed)
    m = X_train.shape[1]
    costs = []  # keep track of cost

    # Parameters initialization.
    if initialization == "he":
        parameters = initialize_parameters_he(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    else:
        parameters = initialize_parameters_zeros(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # If there is no mini-batch size defined, we do batch gradient descent
    if mini_batch_size is None:
        mini_batch_size = m

    # Loop (gradient descent)
    i = 0
    for e in range(num_epochs):
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed += 1
        minibatches = random_mini_batches(X_train, Y_train, mini_batch_size, seed)

        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = L_model_forward(minibatch_X, parameters)

            # Compute cost.
            cost = compute_cost(AL, minibatch_Y)

            # Backward propagation.
            grads = L_model_backward(AL, minibatch_Y, caches)

            # Update parameters
            if optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, learning_rate, beta1, beta2,
                                                               epsilon)
            else:
                parameters = update_parameters(parameters, grads, learning_rate)

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print("Cost after batch %i: %f" % (i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)

            i += 1

    # Predict train set examples and print error
    Y_prediction_train = predict_binary(parameters, X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

    if X_test is not None and Y_test is not None:
        # Predict test set examples and print error
        Y_prediction_test = predict_binary(parameters, X_test)
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    if print_cost:
        # plot the cost
        plot_cost(costs, learning_rate)

    return parameters
