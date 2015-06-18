"""
A module of deep feature selection based on stacked contractive autoencoder.
This module applies a deep structure with many hidden layers. 
Thus, greedy layer-wise pretraining and supervised funetuning are used in optimization.

Copyright (c) 2008-2013, Theano Development Team All rights reserved.

Yifeng Li
CMMT, UBC, Vancouver
Sep 23, 2014
Contact: yifeng.li.cn@gmail.com
"""
from __future__ import division

import time

import math
import copy
import numpy

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from deep_feat_select_mlp import InputLayer
from cA import cA
from gc import collect as gc_collect
import classification as cl



class DFS(object):
    """Deep feature selection class. 
    This structure is input_layer + stacked contractive autoencoder.
    """

    def __init__(self, rng, n_in=784,
                 n_hidden=[500, 500], 
                 n_out=10, activation=T.nnet.sigmoid,
                 lambda1=0,lambda2=0,alpha1=0,alpha2=0,batch_size=100):
        """ Initialize the parameters for the DFL class.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        
        activation: activation function, from {T.tanh, T.nnet.sigmoid (default)}
        
        lambda1: float scalar, control the sparsity of the input weights.
        The regularization term is lambda1( (1-lambda2)/2 * ||w||_2^2 + lambda2 * ||w||_1 ).
        Thus, the larger lambda1 is, the sparser the input weights are.
        
        lambda2: float scalar, control the smoothness of the input weights.
        The regularization term is lambda1( (1-lambda2)/2 * ||w||_2^2 + lambda2 * ||w||_1 ).
        Thus, the larger lambda2 is, the smoother the input weights are.
        
        alpha1: float scalar, control the sparsity of the weight matrices in MLP.
        The regularization term is alpha1( (1-alpha2)/2 * \sum||W_i||_2^2 + alpha2 \sum||W_i||_1 ).
        Thus, the larger alpha1 is, the sparser the MLP weights are.
        
        alpha2: float scalar, control the smoothness of the weight matrices in MLP.
        The regularization term is alpha1( (1-alpha2)/2 * \sum||W_i||_2^2 + alpha2 \sum||W_i||_1 ).
        Thus, the larger alpha2 is, the smoother the MLP weights are.
        
        batch_size: int, minibatch size.
        """

        self.hidden_layers = []
        self.cA_layers = []
        self.params = []
        self.n_layers = len(n_hidden)

        assert self.n_layers > 0

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data, each row is a sample
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        # input layer
        input_layer=InputLayer(input=self.x,n_in=n_in)
        self.params.extend(input_layer.params)
        self.input_layer=input_layer
        # hidden layers
        for i in range(len(n_hidden)):
            if i==0:
                input_hidden=self.input_layer.output
                n_in_hidden=n_in
            else:
                input_hidden=self.hidden_layers[i-1].output
                n_in_hidden=n_hidden[i-1]
            hd=HiddenLayer(rng=rng, input=input_hidden, n_in=n_in_hidden, n_out=n_hidden[i],
                           activation=T.nnet.sigmoid)
            self.hidden_layers.append(hd)
            self.params.extend(hd.params)
            cA_layer = cA(numpy_rng=rng,
                          input=input_hidden,
                          n_visible=n_in_hidden,
                          n_hidden=n_hidden[i],
                          n_batchsize=batch_size,
                          W=hd.W,
                          bhid=hd.b)
            self.cA_layers.append(cA_layer)
            
        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        if len(n_hidden)<=0:
            self.logRegressionLayer = LogisticRegression(
                input=self.input_layer.output,
                n_in=n_in,
                n_out=n_out)
        else:
            self.logRegressionLayer = LogisticRegression(
                input=self.hidden_layers[-1].output,
                n_in=n_hidden[-1],
                n_out=n_out)
        self.params.extend(self.logRegressionLayer.params)
        
        # regularization terms on coefficients of input layer 
        self.L1_input=abs(self.input_layer.w).sum()
        self.L2_input=(self.input_layer.w **2).sum()
        #self.hinge_loss_neg=(T.maximum(0,-self.input_layer.w)).sum() # penalize negative values
        #self.hinge_loss_pos=(T.maximum(0,self.input_layer.w)).sum()  # # penalize positive values
        # regularization terms on weights of hidden layers        
        L1s=[]
        L2_sqrs=[]
        for i in range(len(n_hidden)):
            L1s.append (abs(self.hidden_layers[i].W).sum())
            L2_sqrs.append((self.hidden_layers[i].W ** 2).sum())
        L1s.append(abs(self.logRegressionLayer.W).sum())
        L2_sqrs.append((self.logRegressionLayer.W ** 2).sum())        
        self.L1 = T.sum(L1s)
        self.L2_sqr = T.sum(L2_sqrs)

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors(self.y)
#        self.cost = self.negative_log_likelihood(self.y) \
#         + lambda1*(1.0-lambda2)*0.5*self.L2_input \
#         + lambda1*lambda2*(1.0-lambda3)*self.hinge_loss_pos \
#         + lambda1*lambda2*lambda3*self.hinge_loss_neg \
#         + alpha1*(1.0-alpha2)*0.5 * self.L2_sqr + alpha1*alpha2 * self.L1
        self.cost = self.negative_log_likelihood(self.y) \
         + lambda1*(1.0-lambda2)*0.5*self.L2_input \
         + lambda1*lambda2*self.L1_input \
         + alpha1*(1.0-alpha2)*0.5 * self.L2_sqr + alpha1*alpha2 * self.L1
        self.y_pred=self.logRegressionLayer.y_pred
        self.y_pred_prob=self.logRegressionLayer.y_pred_prob
        
    def get_params(self):
        return copy.deepcopy(self.params)

    def set_params(self, given_params):
        self.params=given_params
    
    def print_params(self):
        for param in self.params:
            print param.get_value(borrow=True)
        
    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the cA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a cA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the cA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the cA layers
        '''

        index = T.lscalar('index')  # index to a minibatch
        contraction_level = T.scalar('contraction_level')  # % of corruption to use
        learning_rate = T.scalar('learning_rate')  # learning rate to use
        # number of batches
        #n_batches = int(math.ceil(train_set_x.get_value(borrow=True).shape[0] / batch_size))
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for cA_layer in self.cA_layers:
            # get the cost and the updates list
            cost, updates = cA_layer.get_cost_updates(contraction_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(contraction_level, default=0.1),
                              theano.Param(learning_rate, default=0.1)],
                                 outputs=[T.mean(cA_layer.L_rec), cA_layer.L_jacob],
                                 updates=updates,
                                 givens={self.x: train_set_x[batch_begin:batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, train_set_x, train_set_y, valid_set_x, valid_set_y, batch_size, learning_rate_shared):
        '''
        Build symbolic funetuning functions for training and validating.
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
        Build a symbolic test function.
        
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

def pretrain_model(model,train_set_x=None,
                   pretrain_lr=0.1,pretraining_epochs=100,
                   batch_size=100,contraction_levels=None):
    """
    Pretrain the model given data in a layer-wise way.
    """
    if not contraction_levels: # if not provided, put it to zeros
        contraction_levels=numpy.zeros((model.n_layers,),dtype=int)
        
    # get the pretraining functions for each layer
    pretraining_fns = model.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size)
    #n_train_batches = int(math.ceil(train_set_x.get_value(borrow=True).shape[0] / batch_size))
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size 
    print '... pretraining the model'
    # pretrain each layer
    for i in xrange(model.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c_batch,j_batch=pretraining_fns[i](index=batch_index,
                                                    contraction_level=contraction_levels[i],
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
    Finetune the model by training and validation sets in a supervised way.
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
                lambda1=0, lambda2=0, alpha1=0, alpha2=0,
                n_hidden=[256,256], contraction_levels=[0.1,0.1],
                pretraining_epochs=20, training_epochs=1000,
                batch_size=100, rng=numpy.random.RandomState(100)):
    """
    Train a model of deep feature selection. 
    
    INPUTS:
    train_set_x_org: numpy 2d array, each row is a training sample.
    
    train_set_y_org: numpy vector of type int {0,1,...,C-1}, class labels of training samples.
    
    valid_set_x_org: numpy 2d array, each row is a validation sample. 
    This set is to monitor the convergence of optimization.
    
    valid_set_y_org: numpy vector of type int {0,1,...,C-1}, class labels of validation samples.
    
    pretrain_lr: float scalar, the learning rate of pretraining phase.
    
    finetune_lr: float scalar, the initial learning rate of finetuning phase.
    
    alpha: float, parameter to trade off the momentum term.
    
    lambda1: float scalar, control the sparsity of the input weights.
    The regularization term is lambda1( (1-lambda2)/2 * ||w||_2^2 + lambda2 * ||w||_1 ).
    Thus, the larger lambda1 is, the sparser the input weights are.
        
    lambda2: float scalar, control the smoothness of the input weights.
    The regularization term is lambda1( (1-lambda2)/2 * ||w||_2^2 + lambda2 * ||w||_1 ).
    Thus, the larger lambda2 is, the smoother the input weights are.
        
    alpha1: float scalar, control the sparsity of the weight matrices in MLP.
    The regularization term is alpha1( (1-alpha2)/2 * \sum||W_i||_2^2 + alpha2 \sum||W_i||_1 ).
    Thus, the larger alpha1 is, the sparser the MLP weights are.
    
    alpha2: float scalar, control the smoothness of the weight matrices in MLP.
    The regularization term is alpha1( (1-alpha2)/2 * \sum||W_i||_2^2 + alpha2 \sum||W_i||_1 ).
    Thus, the larger alpha2 is, the smoother the MLP weights are.
    
    n_hidden, vector of int, n_hidden[i]: number of hidden units of the i-th layer.
    
    contraction_levels: vector of int, contraction_levels[i]: contraction level of the i-th hidden layer 
    
    pretraining_epochs: int scalar, maximal number of epochs in the pretraining phase.
    
    training_epochs: int scalar, maximal number of epochs in the finetuning phase.
    
    batch_size: int scalar, minibatch size.
    
    rng: numpy random number state.
    
    OUTPUTS:
    sca: object of ScA, the model learned, returned for testing.
    
    training_time: float, training time in seconds. 
    """                
    train_set_x = theano.shared(numpy.asarray(train_set_x_org,dtype=theano.config.floatX),borrow=True)
    train_set_y = T.cast(theano.shared(numpy.asarray(train_set_y_org,dtype=theano.config.floatX),borrow=True),'int32')    
    valid_set_x = theano.shared(numpy.asarray(valid_set_x_org,dtype=theano.config.floatX),borrow=True)
    valid_set_y = T.cast(theano.shared(numpy.asarray(valid_set_y_org,dtype=theano.config.floatX),borrow=True),'int32')    
    
    # build the model
    n_feat=train_set_x.get_value(borrow=True).shape[1]
    n_cl=len(numpy.unique(train_set_y_org))
    dfs=DFS(rng=rng, n_in=n_feat,
                 n_hidden=n_hidden, n_out=n_cl, 
                 lambda1=lambda1, lambda2=lambda2, alpha1=alpha1, alpha2=alpha2,batch_size=batch_size)
    # pretrain the model
    start_time=time.clock()
    pretrain_model(dfs,train_set_x,pretrain_lr=pretrain_lr,pretraining_epochs=pretraining_epochs,
                   contraction_levels=contraction_levels,
                   batch_size=batch_size)
    # finetune
    finetune_model(dfs,train_set_x=train_set_x, train_set_y=train_set_y, 
                    valid_set_x=valid_set_x, valid_set_y=valid_set_y, 
                    learning_rate=finetune_lr, alpha=alpha, 
                    n_hidden=n_hidden, n_cl=n_cl, 
                    n_epochs=training_epochs, batch_size=batch_size, rng=rng)
    end_time=time.clock()
    training_time=end_time-start_time
    return dfs, training_time

def test_model(classifier,test_set_x_org,batch_size=200):
    """
    Predict class labels of given data using the model learned.
    
    INPUTS:
    classifier_trained: object of DFS, the model learned by function "train_model". 
    
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
