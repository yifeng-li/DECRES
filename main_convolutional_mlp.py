"""
An example of running convolutional neural network. 

Yifeng Li
CMMT, UBC, Vancouver
Sep. 23, 2014
Contact: yifeng.li.cn@gmail.com
"""
import os
import numpy

import convolutional_mlp
import classification as cl
from gc import collect as gc_collect

numpy.warnings.filterwarnings('ignore')

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

##################################
#load your data here ...
##################################
filename="/home/yifengli/research/dnashape/result/Data_1000bp.txt";
data=numpy.loadtxt(filename,delimiter='\t',dtype='float16')
filename="/home/yifengli/research/dnashape/result/Classes_1000bp.txt";
classes=numpy.loadtxt(filename,delimiter='\t',dtype=str)
filename="/home/yifengli/research/dnashape/result/Features.txt";
features=numpy.loadtxt(filename,delimiter='\t',dtype=str)

# change class labels
given=["Enhancer","EnhancerFalse"]
data,classes=cl.take_some_classes(data,classes,given)

given={"Enhancer":0,"EnhancerFalse":1}
classes=cl.change_class_labels_to_given(classes,given)

train_set_x_org,train_set_y_org,valid_set_x_org,valid_set_y_org,test_set_x_org,test_set_y_org \
=cl.partition_train_valid_test(data,classes,ratio=(1,1,1))    
del data
gc_collect()

rng=numpy.random.RandomState(1000)    
numpy.warnings.filterwarnings('ignore')        
# train
classifier,training_time=convolutional_mlp.train_model( train_set_x_org=train_set_x_org, train_set_y_org=train_set_y_org,
                        valid_set_x_org=valid_set_x_org, valid_set_y_org=valid_set_y_org, 
                        n_row_each_sample=4,
                        learning_rate=0.1, alpha=0.01, n_epochs=1000, rng=rng, 
                        nkerns=[4,4,8],batch_size=500,
                        receptive_fields=((2,8),(2,8),(2,2)),poolsizes=((1,8),(1,8),(1,2)),full_hidden=8)

# test
test_set_y_pred,test_set_y_pred_prob,test_time=convolutional_mlp.test_model(classifier,test_set_x_org)
print test_set_y_pred[0:20]
print test_set_y_pred_prob[0:20]
print test_time
# evaluate classification performance
perf,conf_mat=cl.perform(test_set_y_org,test_set_y_pred,numpy.unique(train_set_y_org))
print perf
print conf_mat

# collect garbage
gc_collect()
