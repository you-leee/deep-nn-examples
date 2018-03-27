from commons.dataload_utils.load_mnist import load_mnist
from commons.datatransform_utils.one_hot import convert_to_one_hot
import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np
from keras.optimizers import Adam
from ConvolutionalNeuralNetworks.resnet50.ResNet50 import ResNet50


if __name__ == '__main__':
    # Load mnist dataset
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_mnist()

    # Example of image
    index = 12
    plt.imshow(X_train_orig[index, :, :, 0], cmap='gray')
    plt.title('Example image showing number {}'.format(Y_train_orig[index][0]))
    plt.show()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 10).T
    Y_test = convert_to_one_hot(Y_test_orig, 10).T

    # Dimensions
    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("Image shape: " + str(X_train.shape[1:3]))
    print("number of classes: " + str(Y_train.shape[1]))


    # Fit the ResNetMnist model
    K.set_image_data_format('channels_last')
    K.set_learning_phase(1)

    model = ResNet50(input_shape=(28, 28, 1), classes=10)
    optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=1, batch_size=128)

    # Get predicition metrics on test data
    preds = model.predict(X_test).argmax(axis=-1)
    labels = Y_test_orig.flatten()
    correct = np.nonzero(preds == labels)[0]
    incorrect = np.nonzero(preds != labels)[0]

    print("Test Accuracy = " + str(len(correct)/X_test.shape[0]))

    # Plot some misclassified examples
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(wspace=0.15, hspace=0.4)
    plt.rcParams.update({'font.size': 7})

    grid_size = 4
    num_plots = min(len(incorrect), grid_size*grid_size)

    for c, i in enumerate(incorrect[0:num_plots]):
        plt.subplot(grid_size, grid_size, c + 1)
        predicted = preds[i]
        actual = labels[i]

        plt.title('Predicted {} - Actual {}'.format(predicted, actual))
        plt.imshow(X_test_orig[i, :, :, 0], cmap='gray')
    plt.show()
