"""
A module of contractive autoencoder modified 
from the Deep Learning Tutorials (www.deeplearning.net/tutorial/).

Copyright (c) 2008-2013, Theano Development Team All rights reserved.

Modified by Yifeng Li
CMMT, UBC, Vancouver
Sep 23, 2014
Contact: yifeng.li.cn@gmail.com
"""
from __future__ import division
import time
import numpy

import theano
import theano.tensor as T

import classification as cl

class cA(object):
    """ Contractive Auto-Encoder class (cA)

    The contractive autoencoder tries to reconstruct the input with an
    additional constraint on the latent space. With the objective of
    obtaining a robust representation of the input space, we
    regularize the L2 norm(Froebenius) of the jacobian of the hidden
    representation with respect to the input. Please refer to Rifai et
    al.,2011 for more details.

    If x is the input then equation (1) computes the projection of the
    input into the latent space h. Equation (2) computes the jacobian
    of h with respect to x.  Equation (3) computes the reconstruction
    of the input, while equation (4) computes the reconstruction
    error and the added regularization term from Eq.(2).

    .. math::

        h_i = s(W_i x + b_i)                                             (1)

        J_i = h_i (1 - h_i) * W_i                                        (2)

        x' = s(W' h  + b')                                               (3)

        L = -sum_{k=1}^d [x_k \log x'_k + (1-x_k) \log( 1-x'_k)]
             + lambda * sum_{i=1}^d sum_{j=1}^n J_{ij}^2                 (4)

    """

    def __init__(self, numpy_rng, input=None, n_visible=784, n_hidden=100,
                 n_batchsize=1, W=None, bhid=None, bvis=None):
        """Initialize the cA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the contraction level. The
        constructor also receives symbolic variables for the input, weights and
        bias.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given
                     one is generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone cA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type n_batchsize int
        :param n_batchsize: number of examples per batch

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
        self.n_batchsize = n_batchsize
        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)),
                                      dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                                   dtype=theano.config.floatX),
                                 borrow=True)

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

        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_jacobian(self, hidden, W):
        """Computes the jacobian of the hidden layer with respect to
        the input, reshapes are necessary for broadcasting the
        element-wise product on the right axis
        """
        return T.reshape(hidden * (1 - hidden),
                         (self.n_batchsize, 1, self.n_hidden)) * T.reshape(
                             W, (1, self.n_visible, self.n_hidden))

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer
        """
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, contraction_level, learning_rate, cost_measure="cross_entropy"):
        """ This function computes the cost and the updates for one trainng
        step of the cA """

        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        J = self.get_jacobian(y, self.W)

        if cost_measure=="cross_entropy":
            #self.L_rec = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
            self.L_rec = T.mean(- T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z),axis=1))
        elif cost_measure=="euclidean":
            self.L_rec = T.mean(T.sum((self.x-z)**2,axis=1)) 
            
        # Compute the jacobian and average over the number of samples/minibatch
        self.L_jacob = T.mean(T.sum(J ** 2) / self.n_batchsize)
        
        cost = self.L_rec + contraction_level * self.L_jacob

        # compute the gradients of the cost of the `cA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)

def train_model(train_set_x_org=None, training_epochs=1000, batch_size=100,
                n_hidden=10,learning_rate=0.1,contraction_level=0.1,
                cost_measure="cross_entropy", rng=numpy.random.RandomState(100)):
    """
    Train a contractive autoencoder. 
    
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
    ca: object of cA, the model learned, returned for testing.
    
    train_set_x_extracted: reduced training set.
    
    training_time: float, training time in seconds. 
    """
    train_set_x = theano.shared(numpy.asarray(train_set_x_org,dtype=theano.config.floatX),borrow=True)
    #train_set_y = T.cast(theano.shared(numpy.asarray(train_set_y_org,dtype=theano.config.floatX),borrow=True),'int32')
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size    
    #n_train_batches = int(math.ceil(train_set_x.get_value(borrow=True).shape[0] / batch_size))
    # shared variable to reduce the learning rate
    learning_rate_shared=theano.shared(learning_rate,name='learn_rate_shared')
#    learning_rate_init=T.scalar(name='learning_rate_init',dtype=theano.config.floatX)
#    epoch_variable=T.iscalar(name='epoch_variable')
    decay_rate=T.scalar(name='decay_rate',dtype=theano.config.floatX)
#    compute_learn_rate=theano.function([learning_rate_init,epoch_variable,decay_rate],learning_rate_shared, \
#    updates=[(learning_rate_shared,learning_rate_init*decay_rate**(epoch_variable//100))]) # thenao does not support math.pow, instead use T.pow() or a**b
    reduce_learning_rate=theano.function([decay_rate],learning_rate_shared,updates=[(learning_rate_shared,learning_rate_shared*decay_rate)])    
    
    n_visible=train_set_x_org.shape[1] # number of input features
    
    # define the model
    x=T.matrix(name='x',dtype=theano.config.floatX) # define a symbol for the input data (training, validation, or test data)
    ca=cA(numpy_rng=rng, input=x, n_visible=n_visible, n_hidden=n_hidden,n_batchsize=batch_size)
    # get the formula of the cost and updates    
    cost,updates=ca.get_cost_updates(contraction_level=contraction_level, learning_rate=learning_rate,
                                     cost_measure=cost_measure) 
    index=T.lscalar() # symbol for the index
    # define a function to update the cost and model parameters using the formula above     
    train_ca_one_iteration=theano.function([index], [ca.L_rec, ca.L_jacob], updates=updates,
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
            c_batch,j_batch=train_ca_one_iteration(batch_index)
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
    extracted=ca.get_hidden_values(train_set_x)
    get_extracted=theano.function([],extracted)
    train_set_x_extracted=get_extracted()
    return ca, train_set_x_extracted, training_time

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
