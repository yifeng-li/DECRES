"""
A module of multilayer perceptrons modified from the Deep Learning Tutorial. 
This implementation is based on Theano and stochastic gradient descent.

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
numpy.warnings.filterwarnings('ignore') # Theano causes some warnings

import theano
import theano.tensor as T

import classification as cl

class LogisticRegression(object):
    """
    Multi-class logistic regression class. 
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1) # labels
        self.y_pred_prob=T.max(self.p_y_given_x,axis=1) # probabilities

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def get_predicted(self,data):
        """
        Get the class labels and probabilities  given data.
        """
        p_y_given_x = T.nnet.softmax(T.dot(data, self.W) + self.b)
        y_pred = T.argmax(p_y_given_x, axis=1)
        y_pred_prob = T.max(p_y_given_x, axis=1)
        return y_pred,y_pred_prob

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
    
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


def train_model(learning_rate=0.1, n_epochs=1000,
                train_set_x_org=None,train_set_y_org=None,valid_set_x_org=None,valid_set_y_org=None,
                           batch_size=100):
    """
    Train the logistic regression model. 
    
    INPUTS:
    learning_rate: float scalar, the initial learning rate.
    
    n_epochs: int scalar, the maximal number of epochs.
    
    train_set_x_org: numpy 2d array, each row is a training sample.
    
    train_set_y_org: numpy vector of type int {0,1,...,C-1}, class labels of training samples.
    
    valid_set_x_org: numpy 2d array, each row is a validation sample. 
    This set is to monitor the convergence of optimization.
    
    valid_set_y_org: numpy vector of type int {0,1,...,C-1}, class labels of validation samples.
    
    batch_size: int scalar, minibatch size.
    
    OUTPUTS:
    classifier: object of logisticRegression, the model learned, returned for testing.
    
    training_time: float, training time in seconds. 
    """
    train_set_x = theano.shared(numpy.asarray(train_set_x_org,dtype=theano.config.floatX),borrow=True)
    train_set_y = T.cast(theano.shared(numpy.asarray(train_set_y_org,dtype=theano.config.floatX),borrow=True),'int32')    
    valid_set_x = theano.shared(numpy.asarray(valid_set_x_org,dtype=theano.config.floatX),borrow=True)
    valid_set_y = T.cast(theano.shared(numpy.asarray(valid_set_y_org,dtype=theano.config.floatX),borrow=True),'int32')    

    # compute number of minibatches for training, validation and testing
    #n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    #n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_train_batches = int(math.ceil(train_set_x.get_value(borrow=True).shape[0] / batch_size))
    n_valid_batches = int(math.ceil(valid_set_x.get_value(borrow=True).shape[0] / batch_size))
    
    # shared variable to reduce the learning rate
    learning_rate_shared=theano.shared(learning_rate,name='learn_rate_shared')
    #learning_rate_init=T.scalar(name='learning_rate_init',dtype=theano.config.floatX)
    #epoch_variable=T.iscalar(name='epoch_variable')
    decay_rate=T.scalar(name='decay_rate',dtype=theano.config.floatX)
    #compute_learn_rate=theano.function([learning_rate_init,epoch_variable,decay_rate],learning_rate_shared, \
    #updates=[(learning_rate_shared,learning_rate_init*decay_rate**(epoch_variable//100))]) # thenao does not support math.pow, instead use T.pow() or a**b
    reduce_learning_rate=theano.function([decay_rate],learning_rate_shared,updates=[(learning_rate_shared,learning_rate_shared*decay_rate)])    
   
   # define the model

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # each row is a sample
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    num_feat=train_set_x.get_value(borrow=True).shape[1]
    #print train_set_y.get_value()
    n_cl=len(numpy.unique(train_set_y_org))
    classifier = LogisticRegression(input=x, n_in=num_feat, n_out=n_cl)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    validate_model2 = theano.function(inputs=[],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x,
                y: valid_set_y})
                
    validate_model3 = theano.function(inputs=[], 
                                      outputs=classifier.y_pred,
                                      givens={x:valid_set_x})         
                
    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model_one_iteration` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model_one_iteration = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    # training the model below
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
    max_num_epoch_change_learning_rate=100 # initial maximal number of epochs to change learning rate
    max_num_epoch_not_improve=3*max_num_epoch_change_learning_rate # max number of epochs without improvmenet to terminate the optimization  
    max_num_epoch_change_rate=0.8 # change to max number of epochs to change learning rate
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
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
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
    return classifier,training_time
    
def test_model(classifier_trained,test_set_x_org):
    """
    Predict class labels of given data using the model learned.
    
    INPUTS:
    classifier_trained: object of logisticRegression, the model learned by function "train_model". 
    
    test_set_x_org: numpy 2d array, each row is a sample whose label to be predicted.

    OUTPUTS:
    y_predicted: numpy int vector, the class labels predicted.
    test_set_y_predicted_prob: numpy float vector, the probabilities.
    test_time: test time in seconds.
    """
    start_time=time.clock()
    test_set_x=theano.shared(numpy.asarray(test_set_x_org,dtype=theano.config.floatX),borrow=True)
    data = T.matrix('data')
    get_y_pred,get_y_pred_prob=classifier_trained.get_predicted(data)
    test_model_func = theano.function(inputs=[data], outputs=[get_y_pred,get_y_pred_prob])
    y_predicted,y_predicted_prob=test_model_func(test_set_x.get_value(borrow=True))
    end_time=time.clock()
    test_time=end_time-start_time
    return y_predicted,y_predicted_prob,test_time
    
