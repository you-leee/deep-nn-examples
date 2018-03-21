
# deep-nn-examples

This repository contains toy examples of shallow and deep neural networks along with convolutional and recurrent neural networks.
It also contains a (not optimal) implementation of base neural networks.
Have fun! :)

## Usage
You can either run the examples from command line or from an IDE. The below examples are for command line usage.

### Setup
First set up the environment by running the setup.py file for installation. It will download all the necessary packages to run the examples.

```
> python setup.py install
```

### Run example

```
> python NeuralNetworks/logisticregression_vs_shallownn.py
```

## List of examples:

This is the list of finished examples.. others will follow!

#### Neaural Networks and Regression
* Simple linear regression
`NeuralNetworks/simple_linear_regression.py`

  Plain and simple implementation of linear regression aming to demonstrate how you can approximate data points, that are close to a linear function (in this example y = 2\*x + 4).
 
* Document classification with word embedding
`NeuralNetworks/doc_classification_apple.py`

  An example on how to lear word embeddings using a neural network. The training data contains text from both Apple Inc. and the apple fruit and the goal is to categorize new text into one of these classes. There is a lot of room for improvement, like getting more training data, filtering stop words better or restricting the vocabulary... Feel free to play around!

* Logistic regression vs shallow neural networks
`NeuralNetworks/logisticregression_vs_shallownn.py`
    
    In this example, the aim is to classify a linearly NOT separable dataset. You can see, how much better you can do with a neural network with 1 hidden layer vs a simply logistic regression model. It also demonstartes the increase of accuracy, when we increase the size of the hidden layer.

* Shallow neural networks vs "deeper" neural networks
`NeuralNetworks/shallownn_vs_deepnn.py`

   Classic image binary classification problem: cat vs non-cat. Two neural networks are trained for the same number of iterations, but one with 3 hidden layers and the other with only 1. You can observe, that despite, that the simpler model can reach the same train accuracy, on the test set, there is a significant difference.

* Hand (number) sign classification with tensorflow
`NeuralNetworks/tf_sign_classification.py`

   A 1 hidden layer neural network is used to classify hand signs to numbers (0-9). It is an example on how to implement a simple model using tensorflow, instead of coding the backpropagation/optimization yourself.
---
#### Convolutional Neural Networks
* Simple linear regression
`ConvolutionalNeuralNetworks/cnn_sign_classification.py`

   This demo uses convolutional (and pooling) layers to address the same problem as in the example above ("Hand (number) sign classification with tensorflow" ). The main advantage of using convolutional layers on images is, that you have much less parameters as with a fully connected layer. For example: If the images are only of size 32x32x3 (32 wide, 32 high, 3 color channels), a single fully-connected neuron in a first hidden layer would have 32\*32\*3 = 3072 weights, whereas a convolutional layer with one 4x4 filter has only 4\*4\*3 = 48.


## References
- Python setup: https://docs.python.org/3/distutils/setupscript.html
- Tensorflow: https://www.tensorflow.org
- Deep learning course: https://www.coursera.org/specializations/deep-learning
- Intuitive explonation of ConvNets: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
- ConvNet CIFAR-10: https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html
- Word embeddings: https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/
