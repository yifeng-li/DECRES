"""
An example of running multilayer perceptrons.
Yifeng Li
CMMT, UBC, Vancouver
Sep. 23, 2014
Contact: yifeng.li.cn@gmail.com
"""

import os
import numpy

import mlp
import classification as cl
from gc import collect as gc_collect

numpy.warnings.filterwarnings('ignore') # Theano causes some warnings  

path="/home/yifengli/prog/my/deep_learning_v1_0/"
os.chdir(path)

# load data
"""
A data set includes three files: 

[1]. A TAB seperated txt file, each row is a sample, each column is a feature. 
No row and columns allowd in the txt file.
If an original sample is a matrix (3-way array), a row of this file is actually a vectorized sample,
by concatnating the rows of the original sample.

[2]. A txt file including the class labels. 
Each row is a string (white space not allowed) as the class label of the corresponding row in [1].

[3]. A txt file including the name of features.
Each row is a string (white space not allowed) as the feature name of the corresponding column in [1].
"""

data_dir="/home/yifengli/prog/my/deep_learning_v1_0/data/"
# train set
filename=data_dir + "GM12878_200bp_Data_3Cl_l2normalized_TrainSet.txt";
train_set_x_org=numpy.loadtxt(filename,delimiter='\t',dtype='float32')
filename=data_dir + "GM12878_200bp_Classes_3Cl_l2normalized_TrainSet.txt";
train_set_y_org=numpy.loadtxt(filename,delimiter='\t',dtype=object)
prev,train_set_y_org=cl.change_class_labels(train_set_y_org)
# valid set
filename=data_dir + "GM12878_200bp_Data_3Cl_l2normalized_ValidSet.txt";
valid_set_x_org=numpy.loadtxt(filename,delimiter='\t',dtype='float32')
filename=data_dir + "GM12878_200bp_Classes_3Cl_l2normalized_ValidSet.txt";
valid_set_y_org=numpy.loadtxt(filename,delimiter='\t',dtype=object)
prev,valid_set_y_org=cl.change_class_labels(valid_set_y_org)
# test set
filename=data_dir + "GM12878_200bp_Data_3Cl_l2normalized_TestSet.txt";
test_set_x_org=numpy.loadtxt(filename,delimiter='\t',dtype='float32')
filename=data_dir + "GM12878_200bp_Classes_3Cl_l2normalized_TestSet.txt";
test_set_y_org=numpy.loadtxt(filename,delimiter='\t',dtype=object)
prev,test_set_y_org=cl.change_class_labels(test_set_y_org)

filename=data_dir + "GM12878_Features_Unique.txt";
features=numpy.loadtxt(filename,delimiter='\t',dtype=object)  

rng=numpy.random.RandomState(1000)

# train
classifier,training_time=mlp.train_model(train_set_x_org=train_set_x_org, train_set_y_org=train_set_y_org, 
                        valid_set_x_org=valid_set_x_org, valid_set_y_org=valid_set_y_org, 
                        learning_rate=0.1, alpha=0.01, lambda_reg=0.00001, alpha_reg=0.5, # alpha_reg from interval [0,1]
                        n_hidden=[256,64,16], n_epochs=1000, batch_size=200, 
                        activation_func="tanh", rng=rng) # activation_func can be either "tanh" or "sigmoid"
                        
# test
test_set_y_pred,test_set_y_pred_prob,test_time=mlp.test_model(classifier, test_set_x_org, batch_size=200)
print test_set_y_pred[0:20]
print test_set_y_pred_prob[0:20]
print test_time
# evaluate classification performance
perf,conf_mat=cl.perform(test_set_y_org,test_set_y_pred,numpy.unique(train_set_y_org))
print perf
print conf_mat

gc_collect()
