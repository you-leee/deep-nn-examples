import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ConvolutionalNeuralNetworks.cnn_funcs.cnn_model import cnn_3L_model, forward_propagation
from commons.dataload_utils.load_sings import load_signs
from commons.datatransform_utils.one_hot import convert_to_one_hot

if __name__ == '__main__':
    np.random.seed(1)
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs()

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

    train_accuracy, test_accuracy, parameters = cnn_3L_model(X_train, Y_train, X_test, Y_test, learning_rate=0.0055,
                                                             num_epochs=200)

    # Example prediction
    index = 6

    sess = tf.Session()
    (m, n_H0, n_W0, n_C0) = X_train.shape
    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0), name="X")
    Z3 = forward_propagation(X, parameters)
    output = tf.argmax(Z3, 1)
    sess.run(tf.global_variables_initializer())
    prediction = sess.run(output, feed_dict={X: X_train_orig[None,index, :, :, :]})

    plt.title("Predicted class: {}".format(prediction))
    plt.imshow(X_train_orig[index])
    plt.show()