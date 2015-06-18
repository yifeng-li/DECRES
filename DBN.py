"""
A module of deep belief network. This module implements a stacked RBM. 
This module is modified from 
the Deep Learning Tutorials (www.deeplearning.net/tutorial/).

Copyright (c) 2008-2013, Theano Development Team All rights reserved.

Modified by Yifeng Li
CMMT, UBC, Vancouver
Sep 23, 2014
Contact: yifeng.li.cn@gmail.com
"""
from __future__ import division

import pickle
import time
import math
import copy
import numpy

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from rbm import RBM
import classification as cl
from gc import collect as gc_collect

class DBN(object):
    """Deep Belief Network
    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """
    def __init__(self, rng, n_in=784, n_hidden=[500, 500], 
                 n_out=10, lambda_reg=0.001, alpha_reg=0.001):
        """This class is made to support a variable number of layers.
    
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                   weights
    
        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_in: int
        :param n_in: dimension of the input to the DBN
    
        :type n_hidden: list of ints
        :param n_hidden: intermediate layers size, must contain
                               at least one value

        :type n_out: int
        :param n_out: dimension of the output of the network
       
        :type lambda_reg: float
        :param lambda_reg: paramter to control the sparsity of weights by l_1 norm.
         The regularization term is lambda_reg( (1-alpha_reg)/2 * ||W||_2^2 + alpha_reg ||W||_1 ).
         Thus, the larger lambda_reg is, the sparser the weights are.
        
        :type alpha_reg: float
        :param alpha_reg: paramter from interval [0,1] to control the smoothness of weights by squared l_2 norm.
         The regularization term is lambda_reg( (1-alpha_reg)/2 * ||W||_2^2 + alpha_reg ||W||_1 ),
         Thus, the smaller alpha_reg is, the smoother the weights are.
        """
    
        self.hidden_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(n_hidden)
    
        assert self.n_layers > 0
    
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data, each row is a sample
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
    
        for i in xrange(self.n_layers):
           # construct the sigmoidal layer
    
            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_in
            else:
                input_size = n_hidden[i - 1]
    
            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.hidden_layers[-1].output
    
            sigmoid_layer = HiddenLayer(rng=rng,
                                       input=layer_input,
                                       n_in=input_size,
                                       n_out=n_hidden[i],
                                       activation=T.nnet.sigmoid)
             # add the layer to our list of layers
            self.hidden_layers.append(sigmoid_layer)
             # its arguably a philosophical question...
             # but we are going to only declare that the parameters of the
             # sigmoid_layers are parameters of the StackedDAA
             # the visible biases in the dA are parameters of those
             # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)
    
             # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=rng,
                            theano_rng=None,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=n_hidden[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)
    
         # We now need to add a logistic layer on top of the MLP
        if self.n_layers>0:
            self.logRegressionLayer = LogisticRegression(
                         input=self.hidden_layers[-1].output,
                         n_in=n_hidden[-1], n_out=n_out)
        else:
            self.logRegressionLayer = LogisticRegression(
                         input=self.x,
                         n_in=input_size, n_out=n_out)
    
        self.params.extend(self.logRegressionLayer.params)
        
        # regularization        
        L1s=[]
        L2_sqrs=[]
        for i in range(self.n_layers):
            L1s.append (abs(self.hidden_layers[i].W).sum())
            L2_sqrs.append((self.hidden_layers[i].W ** 2).sum())
        L1s.append(abs(self.logRegressionLayer.W).sum())
        L2_sqrs.append((self.logRegressionLayer.W ** 2).sum())
        self.L1 = T.sum(L1s)
        self.L2_sqr = T.sum(L2_sqrs)        
    
        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood(self.y)
        self.cost=self.negative_log_likelihood + \
        lambda_reg * ( (1.0-alpha_reg)*0.5* self.L2_sqr +  alpha_reg*self.L1)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logRegressionLayer.errors(self.y)
        self.y_pred = self.logRegressionLayer.y_pred
        self.y_pred_prob=self.logRegressionLayer.y_pred_prob
        
    def pretraining_functions(self, train_set_x, batch_size, persistent_k=15):
        '''
        Build the symbolic pretraining functions to update the parameter in one iteration.
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('learning_rate')  # learning rate to use
        # number of batches
        #n_batches = int(math.ceil(train_set_x.get_value(borrow=True).shape[0] / batch_size))
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm_layer in self.rbm_layers:
            # get the cost and the updates list
            cost, updates = rbm_layer.get_cost_updates(learning_rate, persistent=None, k=persistent_k)
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x: train_set_x[batch_begin:batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, train_set_x, train_set_y, valid_set_x, valid_set_y, batch_size, learning_rate_shared):
        '''
        Build the symbolic finetuning functions to update the parameters in one iteration. 
        Validation function is also defined.
        '''
        # compute number of minibatches for training, validation and testing
        n_valid_batches = int(math.ceil(valid_set_x.get_value(borrow=True).shape[0] / batch_size))
       
        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.cost, self.params)
        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate_shared))

        train_fn = theano.function(inputs=[index],
              outputs=self.cost,
              updates=updates,
              givens={
               self.x: train_set_x[index * batch_size:
                                   (index + 1) * batch_size],
               self.y: train_set_y[index * batch_size:
                                   (index + 1) * batch_size]},
              name='train')

#        test_score_i = theano.function([index], self.errors,
#                 givens={
#                   self.x: test_set_x[index * batch_size:
#                                      (index + 1) * batch_size],
#                   self.y: test_set_y[index * batch_size:
#                                      (index + 1) * batch_size]},
#                      name='test')

        valid_score_i = theano.function([index], self.errors,
              givens={
                self.x: valid_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size:
                                    (index + 1) * batch_size]},
                     name='valid')

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
#        def test_score():
#            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score

    def build_test_function(self, test_set_x, batch_size):
        """
        Build the symbolic test function.
        """
        n_test_batches = int(math.ceil(test_set_x.get_value(borrow=True).shape[0] / batch_size))
        index = T.lscalar('index')  # index to a [mini]batch
        test_score_i = theano.function([index], [self.y_pred,self.y_pred_prob],
              givens={self.x: test_set_x[index * batch_size : (index + 1) * batch_size]},
                     name='test')

        # Create a function that scans the entire test set
        def test_score():
            y_pred=[]
            y_pred_prob=[]
            for i in xrange(n_test_batches):
                label,prob=test_score_i(i)
                y_pred.extend(label)
                y_pred_prob.extend(prob)
            return y_pred,y_pred_prob
        return test_score

    def get_params(self):
        return copy.deepcopy(self.params)

    def set_params(self, given_params):
        self.params=given_params
    
    def print_params(self):
        for param in self.params:
            print param.get_value(borrow=True)
    
    def save_params(self,filename):
        f=open(filename,'w') # remove existing file
        f.close()
        f=open(filename,'a')
        for param in self.params:
            pickle.dump(param.get_value(borrow=True),f)
        f.close()

def read_params(filename):
    f=open(filename,'r')
    params=pickle.load(f)
    f.close()
    return params

def pretrain_model(model,train_set_x=None,
                   pretrain_lr=0.1,pretraining_epochs=100,
                   batch_size=100,persistent_k=15):
    """
    Pretrain the model given training data.
    
    """
    # get the pretraining functions for each layer
    pretraining_fns = model.pretraining_functions(train_set_x=train_set_x, 
                                                  batch_size=batch_size,
                                                  persistent_k=persistent_k)
    n_train_batches = int(math.ceil(train_set_x.get_value(borrow=True).shape[0] / batch_size))
    #n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size 
    print '... pretraining the model'
    # pretrain each layer
    for i in xrange(model.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c_batch=pretraining_fns[i](index=batch_index,
                                                    learning_rate=pretrain_lr)
                c.append(c_batch)
            print 'Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c))
    # no need to return model, as it is passed by reference

def finetune_model(classifier=None,
                train_set_x=None, train_set_y=None, valid_set_x=None, valid_set_y=None, 
                learning_rate=0.1, alpha=0.01,  
                n_hidden=[256,128,16], n_cl=2, 
                n_epochs=1000, batch_size=100, rng=numpy.random.RandomState(100)):
    """
    Finetune the model by training and validation sets.
    """

    # compute number of minibatches for training, validation and testing
    n_train_batches = int(math.ceil(train_set_x.get_value(borrow=True).shape[0] / batch_size))
    #n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    # shared variable to reduce the learning rate
    learning_rate_shared=theano.shared(learning_rate,name='learn_rate_shared')
    decay_rate=T.scalar(name='decay_rate',dtype=theano.config.floatX)
    reduce_learning_rate=theano.function([decay_rate],learning_rate_shared,updates=[(learning_rate_shared,learning_rate_shared*decay_rate)])    

    train_model_one_iteration,validate_model=classifier.build_finetune_functions(train_set_x, train_set_y, 
                                                                   valid_set_x, valid_set_y, 
                                                                   batch_size, learning_rate_shared)
    print '... finetuning'
        # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    max_num_epoch_change_learning_rate=100
    max_num_epoch_not_improve=3*max_num_epoch_change_learning_rate    
    max_num_epoch_change_rate=0.8
    learning_rate_decay_rate=0.8
    epoch_change_count=0
    start_time = time.clock()
    
    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        epoch_change_count=epoch_change_count+1
        if epoch_change_count % max_num_epoch_change_learning_rate ==0:
            reduce_learning_rate(learning_rate_decay_rate)
            max_num_epoch_change_learning_rate= \
            cl.change_max_num_epoch_change_learning_rate(max_num_epoch_change_learning_rate,max_num_epoch_change_rate)
            max_num_epoch_not_improve=3*max_num_epoch_change_learning_rate            
            epoch_change_count=0        
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model_one_iteration(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    num_epoch_not_improve=0
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # save a copy of the currently best model parameter
                    best_model_params=classifier.get_params()

            if patience <= iter:
                done_looping = True
                break
        if this_validation_loss >= best_validation_loss:
            num_epoch_not_improve=num_epoch_not_improve+1
            
        if num_epoch_not_improve>=max_num_epoch_not_improve:
            done_looping = True
            break
    # set the best model parameters
    classifier.set_params(best_model_params)
    end_time = time.clock()
    training_time=end_time-start_time
    print 'Training time: %f' %(training_time/60)
    print 'Optimization complete with best validation score of %f,' %(best_validation_loss * 100.)
            
#def test_model(classifier, test_set_x_org):
#    """
#    test or prediction
#    """
#    test_set_x=theano.shared(numpy.asarray(test_set_x_org,dtype=theano.config.floatX),borrow=True)
#    index = T.lscalar()  # index to a [mini]batch
#    data = T.matrix('data')  # the data is presented as rasterized images
#    get_y_pred=classifier.get_predicted(data)
#    test_model_func = theano.function(inputs=[data], outputs=get_y_pred)
#    y_predicted=test_model_func(test_set_x.get_value(borrow=True))
#    return y_predicted


def train_model(train_set_x_org=None, train_set_y_org=None, 
                valid_set_x_org=None, valid_set_y_org=None, 
                pretrain_lr=0.1,finetune_lr=0.1, alpha=0.01, 
                lambda_reg=0.001, alpha_reg=0.001,
                n_hidden=[256,256], persistent_k=15,
                pretraining_epochs=20, training_epochs=1000,
                batch_size=100, rng=numpy.random.RandomState(100)):
    """
    Train the model using training and validation data.
    
    INPUTS:
    train_set_x_org: numpy 2d array, each row is a training sample.
    
    train_set_y_org: numpy vector of type int {0,1,...,C-1}, class labels of training samples.
    
    valid_set_x_org: numpy 2d array, each row is a validation sample. 
    This set is to monitor the convergence of optimization.
    
    valid_set_y_org: numpy vector of type int {0,1,...,C-1}, class labels of validation samples.
    
    pretrain_lr: float scalar, the learning rate of pretraining phase.
    
    finetune_lr: float scalar, the initial learning rate of finetuning phase.
    
    alpha: float, parameter to trade off the momentum term.
    
    lambda_reg: float, paramter to control the sparsity of weights by l_1 norm.
    The regularization term is lambda_reg( (1-alpha_reg)/2 * ||W||_2^2 + alpha_reg ||W||_1 ).
    Thus, the larger lambda_reg is, the sparser the weights are.
        
    alpha_reg: float, paramter to control the smoothness of weights by squared l_2 norm.
    The regularization term is lambda_reg( (1-alpha_reg)/2 * ||W||_2^2 + alpha_reg ||W||_1 ),
    Thus, the smaller alpha_reg is, the smoother the weights are.
    
    n_hidden, vector of int, n_hidden[i]: number of hidden units of the i-th layer.
    
    persistent_chain_k: length of persistent chain from the last sampling to new sampling.
    
    pretraining_epochs: int scalar, maximal number of epochs in the pretraining phase.
    
    training_epochs: int scalar, maximal number of epochs in the finetuning phase.
    
    batch_size: int scalar, minibatch size.
    
    rng: numpy random number state.
    
    OUTPUTS:
    dbn: object of DBN, the model learned, returned for testing.
    
    training_time: float, training time in seconds.
    """                
    train_set_x = theano.shared(numpy.asarray(train_set_x_org,dtype=theano.config.floatX),borrow=True)
    train_set_y = T.cast(theano.shared(numpy.asarray(train_set_y_org,dtype=theano.config.floatX),borrow=True),'int32')    
    valid_set_x = theano.shared(numpy.asarray(valid_set_x_org,dtype=theano.config.floatX),borrow=True)
    valid_set_y = T.cast(theano.shared(numpy.asarray(valid_set_y_org,dtype=theano.config.floatX),borrow=True),'int32')    
    
    # build the model
    n_feat=train_set_x.get_value(borrow=True).shape[1]
    n_cl=len(numpy.unique(train_set_y_org))
    dbn=DBN(rng=rng, n_in=n_feat,
                 n_hidden=n_hidden, n_out=n_cl, 
                 lambda_reg=lambda_reg, alpha_reg=alpha_reg)
    # pretrain the model
    start_time=time.clock()
    pretrain_model(dbn,train_set_x,pretrain_lr=pretrain_lr,pretraining_epochs=pretraining_epochs,
                   persistent_k=persistent_k,
                   batch_size=batch_size)
    # finetune
    finetune_model(dbn,train_set_x=train_set_x, train_set_y=train_set_y, 
                    valid_set_x=valid_set_x, valid_set_y=valid_set_y, 
                    learning_rate=finetune_lr, alpha=alpha, 
                    n_hidden=n_hidden, n_cl=n_cl, 
                    n_epochs=training_epochs, batch_size=batch_size, rng=rng)
    end_time=time.clock()
    training_time=end_time-start_time
    return dbn, training_time

def test_model(classifier,test_set_x_org,batch_size=200):
    """
    Predict class labels of given data using the model learned.
    
    INPUTS:
    classifier_trained: object of DBN, the model learned by function "train_model". 
    
    test_set_x_org: numpy 2d array, each row is a sample whose label to be predicted.
    
    batch_size: int scalar, batch size, efficient for a very large number of test samples.
    
    OUTPUTS:
    test_set_y_predicted: numpy int vector, the class labels predicted.
    test_set_y_predicted_prob: numpy float vector, the probabilities.
    test_time: test time in seconds.
    """
    start_time=time.clock()
    test_set_x = theano.shared(numpy.asarray(test_set_x_org,dtype=theano.config.floatX),borrow=True)
    test_score=classifier.build_test_function(test_set_x,batch_size=batch_size)
    test_set_y_predicted,test_set_y_predicted_prob=test_score()
    end_time=time.clock()
    test_time=end_time-start_time
    return test_set_y_predicted,test_set_y_predicted_prob,test_time
