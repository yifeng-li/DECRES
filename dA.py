"""
A module of denosing autoencoder modified 
from the Deep Learning Tutorials (www.deeplearning.net/tutorial/).

Copyright (c) 2008-2013, Theano Development Team All rights reserved.

Modified by Yifeng Li
CMMT, UBC, Vancouver
Sep 23, 2014
Contact: yifeng.li.cn@gmail.com
"""
from __future__ import division
import time
import math
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import classification as cl

class dA(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 n_visible=784, n_hidden=500,
                 W=None, bhid=None, bvis=None):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            #print 'Numbers of visible, and hidden units: %d, %d:' % n_visible,n_hidden 
            print n_hidden
            print n_visible            
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                         dtype=theano.config.floatX),
                                 borrow=True) # there is no bias in the input/visable layer

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden,
                                                   dtype=theano.config.floatX),
                                 name='b',
                                 borrow=True)

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """
        Computes the values of the hidden layer. 
        """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """
        Computes the reconstructed input given the values of the
        hidden layer
        """
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate, cost_measure="cross_entropy"):
        """ This function computes the cost and the updates for one trainng
        step of the dA 
        """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        if cost_measure=="cross_entropy":
            L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
            # note : L is now a vector, where each element is the
            #        cross-entropy cost of the reconstruction of the
            #        corresponding example of the minibatch. We need to
            #        compute the average of all these to get the cost of
            #        the minibatch
            cost = T.mean(L)
        elif cost_measure=="euclidean":
            L = T.sum((self.x-z)**2,axis=1)
            cost = T.mean(L)
            
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)

def train_model(train_set_x_org=None, training_epochs=1000, batch_size=100,
                n_hidden=10, learning_rate=0.1, corruption_level=0.1, 
                cost_measure="cross_entropy", rng=numpy.random.RandomState(100)):
    """
    Train a denoising autoencoder. 
    
    INPUTS:
    train_set_x_org: numpy 2d array, each row is a training sample.
    
    training_epochs: int scalar, the maximal number of epochs.
    
    batch_size: int scalar, minibatch size.
    
    n_hidden: int scalar, number of hidden units
    
    learning_rate: float scalar, the initial learning rate.
    
    corruption_level: float from interval [0,1), corruption level.
    
    cost_measure: string from {"cross_entropy", "euclidean"}, measure to compute the restructive cost.    
    
    rng: numpy random number state.
    
    OUTPUTS:
    da: object of dA, the model learned, returned for testing.
    
    train_set_x_extracted: reduced training set.
    
    training_time: float, training time in seconds. 
    """
    
    train_set_x = theano.shared(numpy.asarray(train_set_x_org,dtype=theano.config.floatX),borrow=True)
    #train_set_y = T.cast(theano.shared(numpy.asarray(train_set_y_org,dtype=theano.config.floatX),borrow=True),'int32')
    #n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_train_batches = int(math.ceil(train_set_x.get_value(borrow=True).shape[0] / batch_size))
    # shared variable to reduce the learning rate
    learning_rate_shared=theano.shared(learning_rate,name='learn_rate_shared')
#    learning_rate_init=T.scalar(name='learning_rate_init',dtype=theano.config.floatX)
#    epoch_variable=T.iscalar(name='epoch_variable')
    decay_rate=T.scalar(name='decay_rate',dtype=theano.config.floatX)
#    compute_learn_rate=theano.function([learning_rate_init,epoch_variable,decay_rate],learning_rate_shared, \
#    updates=[(learning_rate_shared,learning_rate_init*decay_rate**(epoch_variable//100))]) # thenao does not support math.pow, instead use T.pow() or a**b
    reduce_learning_rate=theano.function([decay_rate],learning_rate_shared,updates=[(learning_rate_shared,learning_rate_shared*decay_rate)])    
    
    n_visible=train_set_x_org.shape[1] # number of input features
    theano_rng=RandomStreams(rng.randint(2**30)) # random symbol
    
    # define the model
    x=T.matrix(name='x',dtype=theano.config.floatX) # define a symbol for the input data (training, validation, or test data)
    da=dA(numpy_rng=rng, theano_rng=theano_rng, input=x, n_visible=n_visible, n_hidden=n_hidden)
    # get the formula of the cost and updates    
    cost,updates=da.get_cost_updates(corruption_level=corruption_level, learning_rate=learning_rate,
                                        cost_measure=cost_measure) # cost_measure can be either"cross_entropy" or "euclidean"
    index=T.lscalar() # symbol for the index
    # define a function to update the cost and model parameters using the formula above     
    train_da_one_iteration=theano.function([index], cost, updates=updates,
                                           givens={x:train_set_x[index*batch_size:(index+1)*batch_size]})
    
    max_num_epoch_change_learning_rate=100
    max_num_epoch_not_improve=3*max_num_epoch_change_learning_rate    
    max_num_epoch_change_rate=0.8
    learning_rate_decay_rate=0.8
    epoch_change_count=0
    best_cost=numpy.inf
    # train the model using training set
    start_time=time.clock()
    
    for epoch in xrange(training_epochs):
        c=[] # costs of all minibatches of this epoch
        epoch_change_count=epoch_change_count+1
        if epoch_change_count % max_num_epoch_change_learning_rate ==0:
            reduce_learning_rate(learning_rate_decay_rate)
            max_num_epoch_change_learning_rate= \
            cl.change_max_num_epoch_change_learning_rate(max_num_epoch_change_learning_rate,max_num_epoch_change_rate)
            max_num_epoch_not_improve=3*max_num_epoch_change_learning_rate            
            epoch_change_count=0
        for batch_index in xrange(n_train_batches):
            c_batch=train_da_one_iteration(batch_index)
            c.append(c_batch)
        this_cost=numpy.mean(c)
        print 'Training eopch: %d, cost: %f' % (epoch,this_cost)
        if this_cost<best_cost:
            best_cost=this_cost
            num_epoch_not_improve=0
        if this_cost>=best_cost:
            num_epoch_not_improve=num_epoch_not_improve+1
        if num_epoch_not_improve>=max_num_epoch_not_improve:
                break
    end_time=time.clock()
    training_time=end_time-start_time
    print 'Training time: %f' %(training_time/60)
    
    # return the trained model and the reduced training set
    extracted=da.get_hidden_values(train_set_x)
    get_extracted=theano.function([],extracted)
    train_set_x_extracted=get_extracted()
    return da, train_set_x_extracted, training_time

def test_model(model_trained,test_set_x_org=None):
    """
    Reduce the dimensionality of given data.
    
    INPUTS:
    model_trained: object of dA, model learned by "train_model".
    
    test_set_x_org: numpy 2d array, each row is an input sample.
    
    OUTPUTS:
    test_set_x_extracted, numpy 2d array, each row is a reduced sample in the feature space.
    """
    test_set_x=theano.shared(numpy.asarray(test_set_x_org,dtype=theano.config.floatX),borrow=True)
    extracted=model_trained.get_hidden_values(test_set_x)
    get_extracted=theano.function([],extracted)
    test_set_x_extracted=get_extracted()
    return test_set_x_extracted


