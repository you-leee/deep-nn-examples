import matplotlib.pyplot as plt
import numpy as np

from commons.datatransform_utils.one_hot import convert_to_one_hot
from commons.dataload_utils.load_sings import load_signs
from NeuralNetworks.tf_nn_funcs.model import tf_3L_model
from NeuralNetworks.tf_nn_funcs.predict import predict

if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs() # Loading the dataset

    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    # Normalize image vectors
    X_train = X_train_flatten/255.
    X_test = X_test_flatten/255.
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)

    parameters = tf_3L_model(X_train, Y_train, X_test, Y_test, num_epochs=1100)


    # Predict sign with index
    index = 0
    X_pred = X_train[:, [index]]
    y_hat = predict(X_pred, parameters)

    plt.title("Predicted class is {}. Original class is {}".format(y_hat, np.squeeze(Y_train_orig[:, index])))
    plt.imshow(X_train_orig[index])
    plt.show()
