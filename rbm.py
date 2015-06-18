"""
A module of restricted Boltzmann machine (RBM) modified
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

class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(self, input=None, n_visible=784, n_hidden=500, \
        W=None, hbias=None, vbias=None, numpy_rng=None,
        theano_rng=None):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)),
                      dtype=theano.config.floatX)
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(value=numpy.zeros(n_hidden,
                                                    dtype=theano.config.floatX),
                                  name='hbias', borrow=True)

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(value=numpy.zeros(n_visible,
                                                    dtype=theano.config.floatX),
                                  name='vbias', borrow=True)

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        # udpate is a dictionary type, updates of values of shared variables 
        # including model parameters and persistent chain
        [pre_sigmoid_nvs, nv_means, nv_samples,
         pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self.gibbs_hvh,
                    # the None are place holders, saying that
                    # chain_start is the initial state corresponding to the
                    # 6th output
                    outputs_info=[None,  None,  None, None, None, chain_start],
                    n_steps=k)

        # determine gradients on RBM parameters
        # not that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            # update is a dictionary, add the parameter update dictionary items 
            updates[param] = param - gparam * T.cast(lr,
                                                    dtype=theano.config.floatX)
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """

        cross_entropy = T.mean(
                T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                      axis=1))

        return cross_entropy

def train_model(rng=numpy.random.RandomState(100), train_set_x_org=None, n_hidden=100, 
                learning_rate=0.1, training_epochs=100, batch_size=100, persistent_chain_k=15):
    """
    Train a RBM model given training data.
    
    INPUTS:
    rng: numpy random number state.
    
    train_set_x_org: numpy 2d array, each row is a training sample.
    
    n_hidden, int, number of hidden units.
    
    learning_rate: float scalar, the initial learning rate.
    
    training_epochs: int scalar, the maximal number of epochs.
    
    batch_size: int scalar, minibatch size.
    
    persistent_chain_k: length of persistent chain from the last sampling to new sampling.
    
    OUTPUTS:
    rbm: object of RBM. The model learned. 
    
    mean_hidden: numpy 2d array, each row is a reduced training sample.
    
    training_time: training time.
    """
                    
    train_set_x = theano.shared(numpy.asarray(train_set_x_org,dtype=theano.config.floatX),borrow=True)
    n_train_batches = int(math.ceil(train_set_x.get_value(borrow=True).shape[0] / batch_size))
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    # shared variable to reduce the learning rate
    learning_rate_shared=theano.shared(learning_rate,name='learn_rate_shared')
#    learning_rate_init=T.scalar(name='learning_rate_init',dtype=theano.config.floatX)
#    epoch_variable=T.iscalar(name='epoch_variable')
    decay_rate=T.scalar(name='decay_rate',dtype=theano.config.floatX)
#    compute_learn_rate=theano.function([learning_rate_init,epoch_variable,decay_rate],learning_rate_shared, \
#    updates=[(learning_rate_shared,learning_rate_init*decay_rate**(epoch_variable//100))]) # thenao does not support math.pow, instead use T.pow() or a**b
    reduce_learning_rate=theano.function([decay_rate],learning_rate_shared,updates=[(learning_rate_shared,learning_rate_shared*decay_rate)])    
    
    n_visible=train_set_x_org.shape[1] # number of input features
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    # construct the RBM class
    rbm = RBM(input=x, n_visible=n_visible,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate,persistent=persistent_chain,k=persistent_chain_k)
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm_one_iteration = theano.function([index], cost, updates=updates,
           givens={x: train_set_x[index * batch_size:(index + 1) * batch_size]},
           name='train_rbm')

    # optimization, gradient descent
    max_num_epoch_change_learning_rate=100
    max_num_epoch_not_improve=2*max_num_epoch_change_learning_rate    
    max_num_epoch_change_rate=0.8
    epoch_change_count=0
    best_cost=numpy.inf
    # train the model using training set
    start_time=time.clock()
    
    for epoch in xrange(training_epochs):
        c=[] # costs of all minibatches of this epoch
        epoch_change_count=epoch_change_count+1
        if epoch_change_count % max_num_epoch_change_learning_rate ==0:
            reduce_learning_rate(0.5)
            max_num_epoch_change_learning_rate= \
            cl.change_max_num_epoch_change_learning_rate(max_num_epoch_change_learning_rate,max_num_epoch_change_rate)
            max_num_epoch_not_improve=2*max_num_epoch_change_learning_rate            
            epoch_change_count=0
        for batch_index in xrange(n_train_batches):
            c_batch=train_rbm_one_iteration(batch_index)
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
    extracted=rbm.propup(train_set_x)
    get_extracted=theano.function([],extracted)
    pre_activation,mean_hidden=get_extracted()
    return rbm, mean_hidden, training_time

def test_model(model_trained,test_set_x_org=None):
    """
    Get the reduced data using the model learned.
    
    INPUTS:
    model_trained: object of RBM, RBM model learned.
    
    test_set_x_org: numpy 2d array, each row is a sample.
    
    OUTPUTS:
    mean_hidden: numpy 2d array, the reduced data. 
    """
    test_set_x=theano.shared(numpy.asarray(test_set_x_org,dtype=theano.config.floatX),borrow=True)
    extracted=model_trained.propup(test_set_x)
    get_extracted=theano.function([],extracted)
    pre_activation,mean_hidden=get_extracted()
    return mean_hidden

def sample_model(rng,model_trained,test_set_x_org=None,n_chains=20,n_samples=10,sample_gap=1000):
    """
    Sample from the trained RBM given some actual examples to initialize the algorithm.
    
    INPUTS:
    rng: numpy random number state.
    
    model_trained: object of RBM, RBM model learned.
    
    test_set_x_org: numpy 2d array, each row is a actual example.
    
    n_chains: number of Gibbs chains to be sampled indepently.
    
    n_samples: int, number of samples to be taking in each chain. 
    A sample is taken every "sample_gap" steps. 
    
    sample_gap: int, steps of Gibbs sampling before taking samples.
    
    OUTPUTS:
    samples_vis: numpy array of n_samples X n_chains X num_visible_units, 
    sampled samples.
    
    samples_vis_mf: numpy array of n_samples X n_chains X num_visible_units, 
    mean fields of sampled samples.
    """
    test_set_x=theano.shared(numpy.asarray(test_set_x_org,dtype=theano.config.floatX),borrow=True)
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(numpy.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX))

    # sampling
    [presig_hids, hid_mfs, hid_samples, presig_vis,
     vis_mfs, vis_samples], updates =  \
                        theano.scan(model_trained.gibbs_vhv,
                                outputs_info=[None,  None, None, None,
                                              None, persistent_vis_chain],
                                n_steps=sample_gap)

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function([], [vis_mfs[-1], vis_samples[-1]],
                                updates=updates,
                                name='sample_fn')

    # sample n_samples here
    samples_vis=numpy.zeros((n_samples,n_chains,model_trained.n_visible),dtype=test_set_x_org.dtype)
    samples_vis_mf=samples_vis    
    for idx in xrange(n_samples):
        vis_mf, vis_sample = sample_fn()
        samples_vis[idx,:,:]=vis_sample
        samples_vis_mf[idx,:,:]=vis_mf
    return samples_vis, samples_vis_mf
