
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

  Plain and simple implementation of linear regression aming to demonstrate how you can approximate data points, that are close to a linear function (in this example y = 2*x + 4).

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

## References
- Python setup: https://docs.python.org/3/distutils/setupscript.html
- Tensorflow: https://www.tensorflow.org
- Deep learning course: https://www.coursera.org/specializations/deep-learning
