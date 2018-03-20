from NeuralNetworks.dataload_utils.load_cat_images import load_cat_images
from NeuralNetworks.nn_funcs.model import L_layer_model
from NeuralNetworks.nn_funcs.predict import predict_binary
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Loading the data (cat/non-cat)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_cat_images()

    # Number of train/test and pixels
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig[0].shape[0]

    print("Number of training examples: m_train = " + str(m_train))
    print("Number of testing examples: m_test = " + str(m_test))
    print("Height/Width of each image: num_px = " + str(num_px))
    print()
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_set_x shape: " + str(train_set_x_orig.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x shape: " + str(test_set_x_orig.shape))
    print("test_set_y shape: " + str(test_set_y.shape))
    print()

    # Reshape the training and test examples
    n_x = num_px * num_px * 3
    train_set_x_flatten = train_set_x_orig.reshape((m_train, n_x)).T
    test_set_x_flatten = test_set_x_orig.reshape((m_test, n_x)).T

    print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print("test_set_y shape: " + str(test_set_y.shape))
    print()

    # Normalize
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    # Train 2 layer network
    print('Train shallow neural net with only 1 hidden layer')
    layers = [n_x, 7, 1]
    shallow_parameters = L_layer_model(train_set_x, train_set_y, layers, test_set_x, test_set_y, num_epochs=1500,
                                 learning_rate=0.0075, print_cost=True)

    # Train 5 layer model
    print('\nTrain deeper neural net with 3 hidden layers')
    layers = [n_x, 20, 7, 5, 1]
    deep_parameters = L_layer_model(train_set_x, train_set_y, layers, test_set_x, test_set_y, num_epochs=1500,
                                 learning_rate=0.0075, print_cost=True)

    # Example of a prediction
    index = 25
    pred = predict_binary(deep_parameters, train_set_x_flatten[:, index:(index + 1)])
    print("Prediction for picture {}: {}".format(index, np.squeeze(pred)))

    plt.figure(10)
    plt.imshow(train_set_x_orig[index])
    plt.show()
