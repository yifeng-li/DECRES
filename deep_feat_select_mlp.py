"""
A module of deep feature selection based on multilayer perceptrons.
This module applies a deep structure with not too many hidden layers. 
Thus, stochastic gradient descent (back-prop) is used in optimization.

Copyright (c) 2008-2013, Theano Development Team All rights reserved.

Yifeng Li
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
import classification as cl

class InputLayer(object):
    def __init__(self, input, n_in, w=None):
        """
        In the input layer x_i is multiplied by w_i.
        Yifeng Li, in UBC.
        Aug 26, 2014.
        """
        self.input=input
        if w is None:
            w_values = numpy.ones((n_in,), dtype=theano.config.floatX)
#            w_values = numpy.asarray(rng.uniform(
#                    low=0, high=1,
#                    size=(n_in,)), dtype=theano.config.floatX)            
            w = theano.shared(value=w_values, name='w', borrow=True)
        self.w=w
        #u_values = numpy.ones((n_in,), dtype=theano.config.floatX)
        #u = theano.shared(value=u_values, name='u', borrow=True)
        #self.u=u # auxiliary variable for non-negativity
        self.output = self.w * self.input
        #self.params=[w,u]
        self.params=[w]
    def get_predicted(self,data):
        return self.w * data

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh by default.
        Hidden unit activation is thus given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.

        self.activation=activation
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]
        
    def get_predicted(self,data):
        lin_output = T.dot(data, self.W) + self.b
        output = (lin_output if self.activation is None
        else self.activation(lin_output))
        return output

class DFS(object):
    """
    Deep feature selection class. One-one input layer + MLP. 
    """

    def __init__(self, rng, n_in, n_hidden, n_out, x=None, y=None, activation=T.tanh,
                 lambda1=0.001, lambda2=1.0, alpha1=0.001, alpha2=0.0):
        """Initialize the parameters for the DFL class.

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
        
        activation: activation function, from {T.tanh, T.nnet.sigmoid}
        
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
        """
        
        if not x:
            x=T.matrix('x')
        self.x=x
        if not y:
            y=T.ivector('y')
        self.y=y
        
        self.hidden_layers=[]
        self.params=[]
        self.n_layers=len(n_hidden)        
        
        input_layer=InputLayer(input=self.x,n_in=n_in)
        self.params.extend(input_layer.params)
        self.input_layer=input_layer
        for i in range(len(n_hidden)):
            if i==0: # first hidden layer
                hd=HiddenLayer(rng=rng, input=self.input_layer.output, n_in=n_in, n_out=n_hidden[i],
                               activation=activation)
            else:
                hd=HiddenLayer(rng=rng, input=self.hidden_layers[i-1].output, n_in=n_hidden[i-1], n_out=n_hidden[i],
                               activation=activation)
            self.hidden_layers.append(hd)
            self.params.extend(hd.params)
            
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
        
        # regularization terms
        self.L1_input=T.abs_(self.input_layer.w).sum()
        self.L2_input=(self.input_layer.w **2).sum()
        self.hinge_loss_neg=(T.maximum(0,-self.input_layer.w)).sum() # penalize negative values
        self.hinge_loss_pos=(T.maximum(0,self.input_layer.w)).sum()  # # penalize positive values
        L1s=[]
        L2_sqrs=[]
        #L1s.append(abs(self.hidden_layers[0].W).sum())
        for i in range(len(n_hidden)):
            L1s.append (T.abs_(self.hidden_layers[i].W).sum())
            L2_sqrs.append((self.hidden_layers[i].W ** 2).sum())
        L1s.append(T.abs_(self.logRegressionLayer.W).sum())
        L2_sqrs.append((self.logRegressionLayer.W ** 2).sum())        
        self.L1 = T.sum(L1s)
        self.L2_sqr = T.sum(L2_sqrs)

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors(self.y)
#        lambda3=0.5
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
    
    def build_train_function(self, train_set_x, train_set_y, batch_size, alpha, learning_rate_shared):
        """
        Create a function to compute the mistakes that are made by the model.
        """
        index = T.lscalar('index')  # index to a [mini]batch

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
                                      self.y: train_set_y[index * batch_size: (index + 1) * batch_size]},
                                      name='train')
        return train_model_cost

    def build_valid_function(self,valid_set_x, valid_set_y, batch_size):
        """
        Build symbolic validation function.
        """
        n_valid_batches = int(math.ceil(valid_set_x.get_value(borrow=True).shape[0] / batch_size))
        
        index = T.lscalar('index')  # index to a [mini]batch
        valid_error_i = theano.function([index], self.errors,
                                        givens={self.x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                                self.y: valid_set_y[index * batch_size:(index + 1) * batch_size]},
                                        name='valid')

        # Create a function that scans the entire validation set
        def valid_error():
            return [valid_error_i(i) for i in xrange(n_valid_batches)]
        return valid_error
        
    def build_test_function(self, test_set_x, batch_size):
        """
        Build symbolic test function.
        
        """
        n_test_batches = int(math.ceil(test_set_x.get_value(borrow=True).shape[0] / batch_size))
        index = T.lscalar('index')  # index to a [mini]batch
        test_pred_i = theano.function([index], [self.y_pred,self.y_pred_prob],
                                       givens={self.x: test_set_x[index * batch_size : (index + 1) * batch_size]},
                                       name='test')

        # Create a function that scans the entire test set
        def test_pred():
            y_pred=[]
            y_pred_prob=[]
            for i in xrange(n_test_batches):
                label,prob=test_pred_i(i)
                y_pred.extend(label)
                y_pred_prob.extend(prob)
            return y_pred,y_pred_prob
        return test_pred     
    
    def get_predicted(self,data):
        for i in range(len(self.hidden_layers)):
            data=self.hidden_layers[i].get_predicted(data)
        p_y_given_x = T.nnet.softmax(T.dot(data, self.logRegressionLayer.W) + self.logRegressionLayer.b)
        y_pred = T.argmax(p_y_given_x, axis=1)
        y_pred_prob = T.argmax(p_y_given_x, axis=1)
        return y_pred,y_pred_prob
        
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

def train_model(train_set_x_org=None, train_set_y_org=None, valid_set_x_org=None, valid_set_y_org=None, 
                learning_rate=0.1, alpha=0.01, 
                lambda1=0.001, lambda2=1.0, alpha1=0.001, alpha2=0.0, 
                n_hidden=[256,128,16], n_epochs=1000, batch_size=100, 
                activation_func="tanh", rng=numpy.random.RandomState(100),
                max_num_epoch_change_learning_rate=100,max_num_epoch_change_rate=0.8,learning_rate_decay_rate=0.8):
    """
    Train a deep feature selection model. 
    
    INPUTS:
    train_set_x_org: numpy 2d array, each row is a training sample.
    
    train_set_y_org: numpy vector of type int {0,1,...,C-1}, class labels of training samples.
    
    valid_set_x_org: numpy 2d array, each row is a validation sample. 
    This set is to monitor the convergence of optimization.
    
    valid_set_y_org: numpy vector of type int {0,1,...,C-1}, class labels of validation samples.
        
    learning_rate: float scalar, the initial learning rate.
    
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
    
    n_epochs: int scalar, the maximal number of epochs.
    
    batch_size: int scalar, minibatch size.
    
    activation_func: string, specify activation function. {"tanh" (default),"sigmoid"}
    
    rng: numpy random number state.
    
    OUTPUTS:
    classifier: object of MLP, the model learned, returned for testing.
    
    training_time: float, training time in seconds.
    """
    train_set_x = theano.shared(numpy.asarray(train_set_x_org,dtype=theano.config.floatX),borrow=True)
    train_set_y = T.cast(theano.shared(numpy.asarray(train_set_y_org,dtype=theano.config.floatX),borrow=True),'int32')    
    valid_set_x = theano.shared(numpy.asarray(valid_set_x_org,dtype=theano.config.floatX),borrow=True)
    valid_set_y = T.cast(theano.shared(numpy.asarray(valid_set_y_org,dtype=theano.config.floatX),borrow=True),'int32')    

    # compute number of minibatches for training, validation and testing
    n_train_batches = int(math.ceil(train_set_x.get_value(borrow=True).shape[0] / batch_size))

    # shared variable to reduce the learning rate
    learning_rate_shared=theano.shared(learning_rate,name='learn_rate_shared')
    decay_rate=T.scalar(name='decay_rate',dtype=theano.config.floatX)
    reduce_learning_rate=theano.function([decay_rate],learning_rate_shared,updates=[(learning_rate_shared,learning_rate_shared*decay_rate)])    
    
    ## define the model below
    num_feat=train_set_x.get_value(borrow=True).shape[1] # number of features
    n_cl=len(numpy.unique(train_set_y_org)) # number of classes
    
    activations={"tanh":T.tanh,"sigmoid":T.nnet.sigmoid}  
    activation=activations[activation_func]
    
    # build a MPL object    
    classifier = DFS(rng=rng, n_in=num_feat, n_hidden=n_hidden, n_out=n_cl,
                 lambda1=lambda1, lambda2=lambda2, alpha1=alpha1, alpha2=alpha2,
                 activation=activation)
                     
    train_model_one_iteration=classifier.build_train_function(train_set_x, train_set_y, batch_size, 
                                                              alpha, learning_rate_shared)
    validate_model=classifier.build_valid_function(valid_set_x, valid_set_y, batch_size)
    
    print '... training'
        # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    best_validation_loss = numpy.inf
    #max_num_epoch_change_learning_rate=100
    max_num_epoch_not_improve=3*max_num_epoch_change_learning_rate    
    #max_num_epoch_change_rate=0.8
    #learning_rate_decay_rate=0.8
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
    return classifier, training_time
            
def test_model(classifier, test_set_x_org, batch_size):
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
    test_model_func=classifier.build_test_function(test_set_x, batch_size)
    test_set_y_predicted,test_set_y_predicted_prob=test_model_func()
    end_time=time.clock()
    test_time=end_time-start_time
    return test_set_y_predicted,test_set_y_predicted_prob,test_time

