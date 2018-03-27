import numpy as np
import matplotlib.pyplot as plt
from ConvolutionalNeuralNetworks.cnn_funcs.cnn_model import cnn_3L_model
from commons.dataload_utils.load_sings import load_signs
from commons.datatransform_utils.one_hot import convert_to_one_hot

if __name__ == '__main__':
    np.random.seed(1)
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs()

    # Example
    index = 6
    plt.imshow(X_train_orig[index])
    plt.show()
    print("y = " + str(np.squeeze(Y_train_orig[:, index])))

    # Normalization and one-hot encoding
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T
    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    train_accuracy, test_accuracy, parameters = cnn_3L_model(X_train, Y_train, X_test, Y_test, learning_rate=0.006,
                                                             num_epochs=200)
