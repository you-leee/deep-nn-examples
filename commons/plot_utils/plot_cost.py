from . import np
from . import plt

def plot_cost(costs, learning_rate):
    '''
    Plots the cost per batches

    :param costs: cost for each batch
    :param learning_rate: the learning rate used
    '''

    plt.figure(np.random.randint(10))
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('batches (per 100)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()