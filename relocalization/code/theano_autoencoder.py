import cPickle
import gzip
import os
import pdb

import numpy as np
import theano as th
from theano import tensor as T
from numpy import random as rng


class AutoEncoder(object):
    def __init__(self, X, hidden_size, activation_function,
                 output_function):
        # X is the data, an m x n numpy matrix
        # where rows correspond to datapoints
        # and columns correspond to features.
        assert type(X) is np.ndarray
        assert len(X.shape) == 2
        self.X = X
        self.X = th.shared(name='X', value=np.asarray(self.X,
                                                      dtype=th.config.floatX), borrow=True)

        # The config.floatX and borrow=True stuff is to get this to run
        # fast on the gpu. I recommend just doing this without thinking about
        # it until you understand the code as a whole, then learning more
        # about gpus and theano.
        self.n = X.shape[1]
        self.m = X.shape[0]

        # Hidden_size is the number of neurons in the hidden layer, an int.
        assert type(hidden_size) is int
        assert hidden_size > 0
        self.hidden_size = hidden_size
        initial_W = np.asarray(rng.uniform(
            low=-4 * np.sqrt(6. / (self.hidden_size + self.n)),
            high=4 * np.sqrt(6. / (self.hidden_size + self.n)),
            size=(self.n, self.hidden_size)), dtype=th.config.floatX)
        self.W = th.shared(value=initial_W, name='W', borrow=True)
        self.b1 = th.shared(name='b1', value=np.zeros(shape=(self.hidden_size,),
                            dtype=th.config.floatX), borrow=True)
        self.b2 = th.shared(name='b2', value=np.zeros(shape=(self.n,),
                            dtype=th.config.floatX), borrow=True)
        self.activation_function = activation_function
        self.output_function = output_function
                    
    def train(self, n_epochs=100, mini_batch_size=1, learning_rate=0.1):
        index = T.lscalar()
        x = T.matrix('x')
        params = [self.W, self.b1, self.b2]
        hidden = self.activation_function(T.dot(x, self.W) + self.b1)
        output = T.dot(hidden, T.transpose(self.W)) + self.b2
        output = self.output_function(output)
        
        # Use cross-entropy loss.
        L = -T.sum(x*T.log(output) + (1-x)*T.log(1-output), axis=1)
        cost = L.mean()
        updates = []
        
        # Return gradient with respect to W, b1, b2.
        gparams = T.grad(cost, params)
        
        # Create a list of 2 tuples for updates.
        for param, gparam in zip(params, gparams):
            updates.append((param, param-learning_rate*gparam))
        
        # Train given a mini-batch of the data.
        train = th.function(inputs=[index], outputs=[cost], updates=updates,
                            givens={x: self.X[index: index + mini_batch_size, :]})

        import time
        start_time = time.clock()
        for epoch in xrange(n_epochs):
            print "Epoch:", epoch
            for row in xrange(0, self.m, mini_batch_size):
                train(row)
        end_time = time.clock()
        print "Average time per epoch=", (end_time-start_time)/n_epochs

    def get_hidden(self, data):
        x = T.dmatrix('x')
        hidden = self.activation_function(T.dot(x, self.W) + self.b1)
        transformed_data = th.function(inputs=[x], outputs=[hidden])
        return transformed_data(data)

    def get_weights(self):
        return [self.W.get_value(), self.b1.get_value(), self.b2.get_value()]


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # witch row's correspond to an example. target is a
    # numpy.ndarray of 1 dimensions (vector)) that have the same length as
    # the number of rows in the input. It should give the target
    # target to the example with the same index in the input.

    return (train_set, valid_set, test_set)

path = "/home/quadbase/Work/relocalization/data/mnist.pkl.gz"
data = load_data(path)


def plot_first_k_numbers(X, k):
    from matplotlib import pyplot
    m = X.shape[0]
    k = min(m, k)
    j = int(round(k / 10.0))
    
    fig, ax = pyplot.subplots(j, 10)
   
    for i in range(k):
        w = X[i, :]

        w = w.reshape(28, 28)
        ax[i / 10, i % 10].imshow(w, cmap=pyplot.cm.gist_yarg,
                                  interpolation='nearest', aspect='equal')
        ax[i / 10, i % 10].axis('off')

    pyplot.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    pyplot.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off',
        right='off',    # ticks along the top edge are off
        labelleft='off')

    pyplot.show()
    
# def get_max_activations(vec, p=0.8):
#    import scipy.optimize as opt
#    n=size(vec)
#    init=np.zeros(shape=vec.shape)
#    init= vec*(vec>=0)
#    if np.sum(init)==0:
#        init=np.ones(shape=vec.shape)
#    n_init = np.sum(init**2)**0.5
#    init = init/n_init
#
#    def f(x):
#        return -(1-p)*np.dot(vec,x) + p*(np.dot(x,x)-1)**2
#
#    def grad(x):
#            return -(1-p)*np.transpose(vec) + p*2*(np.dot(x,x)-1)*np.transpose(x)
#
#    bounds = [(0,1) for bound in range(n)]
#    loop=True
#    
#    while(loop):
#        x0=rng.randn(n)+init
#        R= opt.fmin_tnc(func=f, x0=x0, fprime=grad, bounds=bounds,messages=0)               
#        if R[2]==0:
#            loop=False
#        else:
#            loop=True
#
#    return R[0]


def m_test(data):
    X = data[0][0]
    activation_function = T.nnet.sigmoid
    output_function = activation_function
    A = AutoEncoder(X, 500, activation_function, output_function)
    A.train(20, 20)
    W = np.transpose(A.get_weights()[0])
    print W.shape
    print W
    pdb.set_trace()
    # plot_first_k_numbers(W, 100)


m_test(data)
