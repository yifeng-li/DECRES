"""
A module of convolutional neural network (CNN) modified
from the Deep Learning Tutorials (www.deeplearning.net/tutorial/).

Copyright (c) 2008-2013, Theano Development Team All rights reserved.

Modified by Yifeng Li
CMMT, UBC, Vancouver
Sep 23, 2014
Contact: yifeng.li.cn@gmail.com
"""
from __future__ import division

import pickle
import copy
import time
import math
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
import classification as cl

numpy.warnings.filterwarnings('ignore')

class LeNetConvPoolLayer(object):
    """
    Pool layer of a convolutional network 
    """
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

class cnn_sub(object):
    """
    The CNN class without classes as output layers.
    """
    def __init__(self,rng, x=None, y=None, batch_size=100, input_size=None,
                 nkerns=[4,4,4], receptive_fields=((1,8),(1,8),(1,8)), poolsizes=((1,8),(1,8),(1,4)),full_hidden=[16],n_out=2):
        """
        
        """
        self.x=x
        self.y=y
        self.layers=[]
        self.params=[]
        for i in range(len(nkerns)):
            receptive_field=receptive_fields[i]            
            if i==0:
                featmap_size_after_downsample=input_size
                layeri_input = self.x.reshape((batch_size, 1, featmap_size_after_downsample[0], featmap_size_after_downsample[1])) # reshape to tensor
                image_shape=(batch_size, 1, featmap_size_after_downsample[0], featmap_size_after_downsample[1])
                filter_shape=(nkerns[i], 1, receptive_field[0], receptive_field[1])
            else:
                layeri_input=self.layers[i-1].output
                image_shape=(batch_size, nkerns[i-1], featmap_size_after_downsample[0], featmap_size_after_downsample[1])
                filter_shape=(nkerns[i], nkerns[i-1], receptive_field[0], receptive_field[1])
                
            
            layeri = LeNetConvPoolLayer(rng=rng, input=layeri_input,
                                        image_shape=image_shape,
                                        filter_shape=filter_shape, poolsize=poolsizes[i])
            featmap_size_after_conv=get_featmap_size_after_conv(featmap_size_after_downsample,receptive_fields[i])
            featmap_size_after_downsample=get_featmap_size_after_downsample(featmap_size_after_conv,poolsizes[i])
            self.layers.append(layeri)
            self.params.extend(layeri.params)
        
        # multiple fully connected layer
        print 'going to fully connected layers'
        #layer_full_input = self.layers[-1].output.flatten(2)
        # construct a fully-connected sigmoidal layer
        #layer_full = HiddenLayer(rng=rng, input=layer_full_input, 
        #                         n_in=nkerns[-1] * featmap_size_after_downsample[0] * featmap_size_after_downsample[1],
        #                         n_out=full_hidden[0], activation=T.tanh)
        #self.layers.append(layer_full)
        #self.params.extend(layer_full.params)
        #self.output=layer_full.output

        for i in range(len(full_hidden)):
            if i==0:
                layer_full_i_input = self.layers[-1].output.flatten(2) # the output of the last conv-pool layer
                n_i_in = nkerns[-1] * featmap_size_after_downsample[0] * featmap_size_after_downsample[1]
                n_i_out=full_hidden[i]
            else:
                layer_full_i_input=layer_full_i_output
                n_i_in=full_hidden[i-1]
                n_i_out=full_hidden[i]
            
            layer_full_i = HiddenLayer(rng=rng, input=layer_full_i_input, 
                                       n_in=n_i_in, n_out=n_i_out, activation=T.tanh)
            self.layers.append(layer_full_i)
            self.params.extend(layer_full_i.params)
            layer_full_i_output=layer_full_i.output
        self.output=layer_full_i_output

        # construct an output layer (classes) for pretraining
        print 'going to output layer'
        self.logRegressionLayer = LogisticRegression(input=self.layers[-1].output, n_in=full_hidden[-1], n_out=n_out)
        self.params_pretrain=self.params # in fact, self.params_pretrain is the same as self.params
        self.params_pretrain.extend(self.logRegressionLayer.params) # will also extend to self.params
        
        # the cost we minimize during training is the NLL of the model
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood(self.y)
        self.cost = self.logRegressionLayer.negative_log_likelihood(self.y)
        self.errors = self.logRegressionLayer.errors(self.y)
        self.y_pred = self.logRegressionLayer.y_pred
        self.y_pred_prob = self.logRegressionLayer.y_pred_prob
        
class cnn(object):
    """
    The CNN class.
    """
    def __init__(self,rng, batch_size=100, input_size=None,
                 nkerns=[4,4,4], receptive_fields=((1,8),(1,8),(1,8)), poolsizes=((1,8),(1,8),(1,4)),full_hidden_sub=[16], full_hidden_all=[16], n_out=10):
        """
        # input_size: (length of 1d signal, number of 1d signals in a sample)
        """
        self.x = T.matrix(name='x',dtype=theano.config.floatX)   # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
        self.batch_size = theano.shared(value=batch_size,name='batch_size')#T.lscalar('batch_size')

        self.num_cnn=input_size[0] # number of 1d signals, number of sub-CNNs
        len_1d_signal=input_size[1] # length of the 1d signals
        self.layers=[]
        self.params=[]
        self.subcnns=[]
        layer_full_i_input = [] # input of the hidden layer connecting all sub-CNNs

        input=self.x.reshape((batch_size,1,input_size[0],input_size[1])) # reshape x to tensor
        for i in range(self.num_cnn):
            input_sub=input[:,:,i,:] # take the ith signal
            cnn_i=cnn_sub(rng, x=input_sub, y=self.y, batch_size=batch_size, input_size=(1,len_1d_signal),
                          nkerns=nkerns, receptive_fields=receptive_fields, poolsizes=poolsizes,full_hidden=full_hidden_sub,n_out=n_out)
            self.layers.append(cnn_i.layers)
            self.params.extend(cnn_i.params[0:-2]) # exclude the parameters {W,b} of the class layer of a sub-CNN
            self.subcnns.append(cnn_i)
            layer_full_i_input.append(cnn_i.output)

        # multiple fully connected layer
        print 'going to fully connected layers'
        # construct a fully-connected sigmoidal layer to connect all sub CNNs
        #layer_full_i_input=T.concatenate(layer_full_i_input,axis=1) # tensor type
        #layer_full = HiddenLayer(rng=rng, input=layer_full_i_input, 
        #                         n_in=full_hidden_sub[-1]*self.num_cnn,
        #                         n_out=full_hidden_all[0], activation=T.tanh)
        #self.layers.append(layer_full)
        #self.params.extend(layer_full.params)

        for i in range(len(full_hidden_all)):
            if i==0:
                layer_full_i_input = T.concatenate(layer_full_i_input,axis=1) # tensor type, concatnating all outputs of sub-CNNs
                n_i_in = full_hidden_sub[-1]*self.num_cnn
                n_i_out=full_hidden_all[i]
            else:
                layer_full_i_input=layer_full_i_output
                n_i_in=full_hidden_all[i-1]
                n_i_out=full_hidden_all[i]

            layer_full_i = HiddenLayer(rng=rng, input=layer_full_i_input, 
                                       n_in=n_i_in, n_out=n_i_out, activation=T.tanh)
            self.layers.append(layer_full_i)
            self.params.extend(layer_full_i.params)
            layer_full_i_output=layer_full_i.output
        
        # classify the values of the fully-connected sigmoidal layer
        print 'going to output layer'
        self.logRegressionLayer = LogisticRegression(input=self.layers[-1].output, n_in=full_hidden_all[-1], n_out=n_out)
        self.params.extend(self.logRegressionLayer.params)

        # regularization term, L2,1-norm
        L21=self.

            
        
        # the cost we minimize during training is the NLL of the model
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood(self.y)
        self.cost = self.logRegressionLayer.negative_log_likelihood(self.y)
        self.errors = self.logRegressionLayer.errors(self.y)
        self.y_pred = self.logRegressionLayer.y_pred
        self.y_pred_prob = self.logRegressionLayer.y_pred_prob

    def pretraining_functions(self, train_set_x, train_set_y, alpha, batch_size):
        ''' Generates a list of functions, each of them implementing one
        component (sub-CNN) in trainnig the iCNN.
        The function will require as input the minibatch index, and to train
        a sub-CNN you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the sub-CNN

        : train_set_y: ...

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        '''

        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('learning_rate')  # learning rate to use
        # number of batches
        #n_batches = int(math.ceil(train_set_x.get_value(borrow=True).shape[0] / batch_size))
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for subcnn in self.subcnns:
            # create a function to compute the mistakes that are made by the model
            index = T.lscalar('index')  # index to a [mini]batch
            #batch_size_var = T.lscalar('batch_size_var')  # batch_size
            # compute the gradients with respect to the model parameters
            grads = T.grad(subcnn.cost, subcnn.params_pretrain)
        
            # add momentum
            # initialize the delta_i-1
            delta_before=[]
            for param_i in subcnn.params:
                delta_before_i=theano.shared(value=numpy.zeros(param_i.get_value().shape))
                delta_before.append(delta_before_i)
        
            updates = []
            for param_i, grad_i, delta_before_i in zip(subcnn.params, grads, delta_before):
                delta_i=-learning_rate * grad_i + alpha*delta_before_i
                updates.append((param_i, param_i + delta_i ))
                updates.append((delta_before_i,delta_i))
            # compile the theano function
            fn = theano.function([index,theano.Param(learning_rate, default=0.1)], [subcnn.cost,subcnn.errors], updates=updates,
                                      givens={
                                      self.x: train_set_x[index*batch_size:(index+1)*batch_size],
                                      self.y: train_set_y[index*batch_size:(index+1)*batch_size]})
            
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns
                
    def build_train_function(self, train_set_x, train_set_y, batch_size, alpha, learning_rate_shared):
        """
        Build the symbolic training function to update the parameter in one iteration.
        """
        # create a function to compute the mistakes that are made by the model
        index = T.lscalar('index')  # index to a [mini]batch
        #batch_size_var = T.lscalar('batch_size_var')  # batch_size
        # compute the gradients with respect to the model parameters
        grads = T.grad(self.cost, self.params)
        
        # add momentum
        # initialize the delta_i-1
        delta_before=[]
        for param_i in self.params:
            delta_before_i=theano.shared(value=numpy.zeros(param_i.get_value().shape))
            delta_before.append(delta_before_i)
        
        updates = []
        for param_i, grad_i, delta_before_i in zip(self.params, grads, delta_before):
            delta_i=-learning_rate_shared * grad_i + alpha*delta_before_i
            updates.append((param_i, param_i + delta_i ))
            updates.append((delta_before_i,delta_i))
            
        train_model_cost = theano.function([index], self.cost, updates=updates,
                                      givens={
                                      self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                      self.y: train_set_y[index * batch_size: (index + 1) * batch_size]})
        return train_model_cost

    def build_valid_function(self,valid_set_x, valid_set_y, batch_size):
        """
        Build the symbolic validation function to get the validation error.
        """
        n_valid=valid_set_x.get_value(borrow=True).shape[0] # number of validation samples
        n_valid_batches = n_valid// batch_size#int(math.ceil( n_valid/ batch_size))
        
        index = T.lscalar('index')  # index to a [mini]batch
        #batch_size_var = T.lscalar('batch_size_var')  # batch_size
        valid_error_i = theano.function([index], self.errors,
                                        givens={self.x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                                self.y: valid_set_y[index * batch_size:(index + 1) * batch_size]},
                                        name='valid')

        # Create a function that scans the entire validation set
        def valid_error():
            return [valid_error_i(i) for i in xrange(n_valid_batches)]
#            errors=[]
#            for i in xrange(n_valid_batches):
#                if i==n_valid_batches-1:
#                    batch_size_current= n_valid - i*batch_size
#                else:
#                    batch_size_current=batch_size
#                errors.extend(valid_error_i(i,batch_size_current))
#            return errors
        return valid_error
        
    def build_test_function(self, test_set_x):
        """
        Build the symbolic test function to get predicted class labels.
        """
        n_test=test_set_x.get_value(borrow=True).shape[0]
        batch_size=self.batch_size.get_value(borrow=True)
        n_test_batches = n_test//batch_size #int(math.ceil(n_test / batch_size))
        index = T.lscalar('index')  # index to a [mini]batch
#        batch_size_var = T.lscalar('batch_size_var')  # batch_size
#        test_pred_i = theano.function([index,batch_size_var], self.y_pred,
#                                       givens={self.x: test_set_x[index * batch_size_var : (index + 1) * batch_size_var],
#                                               self.batch_size: batch_size_var},
#                                       name='test')

        test_pred_i = theano.function([index], [self.y_pred,self.y_pred_prob],
                                       givens={self.x: test_set_x[index * batch_size : (index + 1) * batch_size]},
                                       name='test')
        test_pred_last = theano.function([], [self.y_pred,self.y_pred_prob],
                                       givens={self.x: test_set_x[-batch_size:]},
                                       name='test')

        # Create a function that scans the entire test set
        def test_pred():
            y_pred=[]
            y_pred=numpy.array(y_pred)
            y_pred_prob=[]
            y_pred_prob=numpy.array(y_pred_prob)
            for i in xrange(n_test_batches):
#                if i==n_test_batches-1:
#                    batch_size_current=n_test - i*batch_size
#                else:
#                    batch_size_current=batch_size    
#                y_pred.extend(test_pred_i(i,batch_size_current))
                label,prob=test_pred_i(i)
                y_pred=numpy.append(y_pred,label)
                y_pred_prob=numpy.append(y_pred_prob,prob)
            left_over=n_test % batch_size
            if left_over >0:
                label,prob= test_pred_last()
                y_pred=numpy.append(y_pred,label[-left_over:])
                y_pred_prob=numpy.append(y_pred_prob,prob[-left_over:])
            return y_pred,y_pred_prob
        return test_pred

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

def get_featmap_size_after_downsample(featmap_size,poolsize):
    featmap_size_after_downsample=featmap_size//poolsize
    return featmap_size_after_downsample
    
def get_featmap_size_after_conv(input_size,receptive_field_size):
    return numpy.array(input_size)-numpy.array(receptive_field_size)+1 

def pretrain_model(model,train_set_x=None,train_set_y=None,
                   pretrain_lr=0.1,pretraining_epochs=100, alpha=0.1,
                   batch_size=100):
    """
    Pretrain the model given training set.
    """
        
    # get the pretraining functions for each layer
    pretraining_fns = model.pretraining_functions(train_set_x=train_set_x, train_set_y=train_set_y, alpha=alpha, batch_size=batch_size)
    #n_train_batches = int(math.ceil(train_set_x.get_value(borrow=True).shape[0] / batch_size))
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size 
    print '... pretraining the model'
    # pretrain each layer
    for i in xrange(model.num_cnn):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = [] # cost
            er = [] # error
            for batch_index in xrange(n_train_batches):
                c_batch,er_batch=pretraining_fns[i](index=batch_index, learning_rate=pretrain_lr)
                c.append(c_batch)
                er.append(er_batch)
            print 'Pre-training sub-cnn %i, epoch %d, cost %f, error: %4f' % (i, epoch, numpy.mean(c), numpy.mean(er))
    # no need to return model, as it is passed by reference

def finetune_model(classifier=None,
                   train_set_x=None,train_set_y=None,valid_set_x=None,valid_set_y=None,
                   n_row_each_sample=1,
                   learning_rate=0.1, alpha=0.1, n_epochs=1000, rng=numpy.random.RandomState(1000), 
                   nkerns=[4,4,4],batch_size=500,
                   receptive_fields=((1,8),(1,8),(1,8)),poolsizes=((1,8),(1,8),(1,4)),full_hidden_sub=[16],full_hidden_all=[16],
                   max_num_epoch_change_learning_rate=80,
                   max_num_epoch_change_rate=0.8,
                   learning_rate_decay_rate=0.8):
    """
    Finetune the model using training and validation data.
    """        
    n_train=train_set_x.get_value(borrow=True).shape[0]
    n_train_batches=n_train//batch_size
    #n_train_batches =  int(math.floor(train_set_x.get_value(borrow=True).shape[0] / batch_size))
    #n_valid_batches = int(math.ceil(valid_set_x.get_value(borrow=True).shape[0] / batch_size))
    
    # shared variable to reduce the learning rate
    learning_rate_shared=theano.shared(learning_rate,name='learn_rate_shared')
#    learning_rate_init=T.scalar(name='learning_rate_init',dtype=theano.config.floatX)
#    epoch_variable=T.iscalar(name='epoch_variable')
    decay_rate=T.scalar(name='decay_rate',dtype=theano.config.floatX)
#    compute_learn_rate=theano.function([learning_rate_init,epoch_variable,decay_rate],learning_rate_shared, \
#    updates=[(learning_rate_shared,learning_rate_init*decay_rate**(epoch_variable//100))]) # thenao does not support math.pow, instead use T.pow() or a**b
    reduce_learning_rate=theano.function([decay_rate],learning_rate_shared,updates=[(learning_rate_shared,learning_rate_shared*decay_rate)])    
    
    train_model_one_iteration=classifier.build_train_function(train_set_x, train_set_y, batch_size, alpha, learning_rate_shared)
    validate_model=classifier.build_valid_function(valid_set_x, valid_set_y, batch_size)
                                                              
    ###############
    # TRAIN MODEL #
    ###############
    print '... finetuning'
    # early-stopping parameters
    #max_num_epoch_change_learning_rate=100
    max_num_epoch_not_improve=5*max_num_epoch_change_learning_rate    
    #max_num_epoch_change_rate=0.8
    #learning_rate_decay_rate=0.8
    epoch_change_count=0
    
    patience = 1000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = n_train_batches; # min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    start_time = time.clock()

    epoch = 0
    done_looping = False
    while (epoch < n_epochs) and (not done_looping): # for every epoch
        epoch = epoch + 1
        epoch_change_count=epoch_change_count+1
        if epoch_change_count % max_num_epoch_change_learning_rate ==0:
            reduce_learning_rate(learning_rate_decay_rate)
            max_num_epoch_change_learning_rate= \
            cl.change_max_num_epoch_change_learning_rate(max_num_epoch_change_learning_rate,max_num_epoch_change_rate)
            max_num_epoch_not_improve=3*max_num_epoch_change_learning_rate            
            epoch_change_count=0
        #compute_learn_rate(learning_rate,epoch,0.5)
        print 'The current learning rate is ', learning_rate_shared.get_value()
        for minibatch_index in xrange(n_train_batches): # for every minibatch

            iter = (epoch - 1) * n_train_batches + minibatch_index # number of total minibatchs so far

            #if iter % 100 == 0:
                #print 'training @ iter = ', iter
            
#            if minibatch_index==n_train_batches-1:
#                batch_size_current=n_train - minibatch_index*batch_size
#            else:
#                batch_size_current=batch_size
#            cost_ij = train_model_one_iteration(minibatch_index,batch_size_current)
            cost_ij = train_model_one_iteration(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %0.4f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    num_epoch_not_improve=0
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
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
    training_time=end_time -start_time
    print 'Finetuning time: %f' %(training_time/60)
    print 'Optimization complete with best validation score of %f,' %(best_validation_loss * 100.)
    
def train_model(train_set_x_org=None,train_set_y_org=None,valid_set_x_org=None,valid_set_y_org=None,
                n_row_each_sample=1,
                pretrain_lr=0.1,finetune_lr=0.1, alpha=0.1, pretraining_epochs=20, training_epochs=1000, rng=numpy.random.RandomState(1000), 
                nkerns=[4,4,4],batch_size=500,
                receptive_fields=((1,8),(1,8),(1,8)),poolsizes=((1,8),(1,8),(1,4)),full_hidden_sub=[16],full_hidden_all=[16],
                max_num_epoch_change_learning_rate=80,
                max_num_epoch_change_rate=0.8,
                learning_rate_decay_rate=0.8):
    """
    Train the model using training and validation data.
    
    INPUTS:
    train_set_x_org: numpy 2d array, each row is a training sample.
    
    train_set_y_org: numpy vector of type int {0,1,...,C-1}, class labels of training samples.
    
    valid_set_x_org: numpy 2d array, each row is a validation sample. 
    This set is to monitor the convergence of optimization.
    
    valid_set_y_org: numpy vector of type int {0,1,...,C-1}, class labels of validation samples.
    
    n_row_each_sample: int, for each vectorized sample, the number of rows when matricize it.
    The vectorized sample is in the form of [row_0,row_1,...,row_{n_row_each_sample-1}].
    
    pretrain_lr: float scalar, the learning rate of pretraining phase.
    
    finetune_lr: float scalar, the initial learning rate of finetuning phase.
    
    alpha: float, parameter to trade off the momentum term.
    
    pretraining_epochs: int scalar, maximal number of epochs in the pretraining phase.
    
    training_epochs: int scalar, maximal number of epochs in the finetuning phase.
    
    rng: numpy random number state.
    
    nkerns: list, tuple, or vector, nkerns[i] is the number of feature maps in the i-th convolutional layer
    
    batch_size: int, minibatch size.
    
    receptive_fields: list or tuple of the same length as nkerns, 
    receptive_fields[i] is a list or tuple of length 2, the size of receptive field in the i-th convolutional layer. 
    receptive_fields[i]= (#rows of the receptive field, #columns of the receptive field).
    
    poolsizes: list or tuple of the same length as nkerns, the size to reduce to scalar. 
    poolsizes[i]=(#rows, #columns)
    
    full_hidden: the number of hidden units fulling connecting the units in the previous layer. 
    
    OUTPUTS:
    classifier: object of CNN class, the model trained.
    
    training_time: training time.
    """    
    train_set_x = theano.shared(numpy.asarray(train_set_x_org,dtype=theano.config.floatX),borrow=True)
    train_set_y = T.cast(theano.shared(numpy.asarray(train_set_y_org,dtype=theano.config.floatX),borrow=True),'int32')    
    valid_set_x = theano.shared(numpy.asarray(valid_set_x_org,dtype=theano.config.floatX),borrow=True)
    valid_set_y = T.cast(theano.shared(numpy.asarray(valid_set_y_org,dtype=theano.config.floatX),borrow=True),'int32')        
    

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    
    num_feat=train_set_x.get_value(borrow=True).shape[1]
    input_size_row=n_row_each_sample # how many rows for each sample
    input_size_col=num_feat//n_row_each_sample
    input_size=(input_size_row,input_size_col)
    n_out=len(numpy.unique(train_set_y_org)) # number of classes
    classifier=cnn(rng=rng, batch_size=batch_size, input_size=input_size,
                 nkerns=nkerns, receptive_fields=receptive_fields, poolsizes=poolsizes,
                   full_hidden_sub=full_hidden_sub, full_hidden_all=full_hidden_all, n_out=n_out)

    start_time = time.clock()
    # pretraining
    pretrain_model(classifier,
                   train_set_x=train_set_x, train_set_y=train_set_y, pretrain_lr=pretrain_lr,
                   pretraining_epochs=pretraining_epochs, alpha=alpha, batch_size=batch_size)
    # finetuning
    finetune_model(classifier,
                   train_set_x=train_set_x,train_set_y=train_set_y,valid_set_x=valid_set_x,valid_set_y=valid_set_y,
                   n_row_each_sample=n_row_each_sample,
                   learning_rate=finetune_lr, alpha=alpha, n_epochs=training_epochs, rng=rng, 
                   nkerns=nkerns, batch_size=batch_size,
                   receptive_fields=receptive_fields,poolsizes=poolsizes,full_hidden_sub=full_hidden_sub,full_hidden_all=full_hidden_all,
                   max_num_epoch_change_learning_rate=max_num_epoch_change_learning_rate,
                   max_num_epoch_change_rate=max_num_epoch_change_rate,
                   learning_rate_decay_rate=learning_rate_decay_rate)
    
    end_time = time.clock()
    training_time=end_time -start_time
    print 'Training time: %f' %(training_time/60)
    return classifier, training_time

def test_model(classifier,test_set_x_org):
    """
    Predict class labels of given data using the model learned.
    
    INPUTS:
    classifier: object of logisticRegression, the model learned by function "train_model". 
    
    test_set_x_org: numpy 2d array, each row is a sample whose label to be predicted.
    
    #batch_size: int scalar, batch size, efficient for a very large number of test samples.
    
    OUTPUTS:
    test_set_y_predicted: numpy int vector, the class labels predicted.
    test_set_y_predicted_prob: numpy float vector, the probabilities.
    test_time: test time in seconds.
    """
    start_time=time.clock()
    test_set_x = theano.shared(numpy.asarray(test_set_x_org,dtype=theano.config.floatX),borrow=True)
    test_model_func=classifier.build_test_function(test_set_x)
    test_set_y_predicted,test_set_y_predicted_prob=test_model_func()
    test_set_y_predicted=numpy.asarray(test_set_y_predicted,dtype=int)
    end_time=time.clock()
    test_time=end_time-start_time
    return test_set_y_predicted,test_set_y_predicted_prob,test_time
