
"""
An example of running deep feature selection (based on DBN) 
for many number of hidden layers.

@author: yifeng
"""
import os
import numpy
import deep_feat_select_DBN
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
lambda1s=[0.005]#numpy.arange(0.0700,-0.001,-0.001)
params_init=None
features_selected=[]
weights_selected=[]
weights=[]
perfs=[]
for i in range(len(lambda1s)):       
    classifier,training_time=deep_feat_select_DBN.train_model(train_set_x_org=train_set_x_org, train_set_y_org=train_set_y_org, 
                valid_set_x_org=valid_set_x_org, valid_set_y_org=valid_set_y_org, 
                pretrain_lr=0.01,finetune_lr=0.1, alpha=0.01, 
                lambda1=lambda1s[i], lambda2=1, alpha1=0.00001, alpha2=0.5,
                n_hidden=[128,64], persistent_k=15,
                pretraining_epochs=5, training_epochs=1000,
                batch_size=200, rng=rng)
 
    param0=classifier.params[0].get_value()
    param1=classifier.params[1].get_value()
    selected=abs(param0)>numpy.max(abs(param0))*0.001
    #selected=abs(param0)>0.001
    features_selected.append(features[selected])
    weights_selected.append(param0[selected])
    print 'Number of select variables:', sum(selected)
    #print features[selected]
    #print  param0[selected]
    weights.append(param0)

    # test
    #test_set_y_pred=dl.test_model(classifier, test_set_x_org)
    test_set_y_pred,test_set_y_pred_prob,test_time=deep_feat_select_DBN.test_model(classifier, test_set_x_org, batch_size=200)
    print test_set_y_pred[0:20]
    print test_set_y_pred_prob[0:20]
    print test_time
    perf,conf_mat=cl.perform(test_set_y_org,test_set_y_pred,numpy.unique(train_set_y_org))
    perfs.append(perf)
    print perf
    print conf_mat
# save result to txt file
#os.makedirs('result')
#filename='./result/GM12878_3Cl_feature_weight_shallow5.txt'
#cl.write_feature_weight(weights,features,lambda1s,filename)
#filename='./result/GM12878_3Cl_feature_weight_unique_selected_shallow5_1e-3.txt'
#perfs=numpy.asarray(perfs)
#cl.write_feature_weight2(weights,features,lambda1s,perfs[:,-1],uniqueness=True,tol=1e-3,filename=filename)
#filename='./result/GM12878_3Cl_feature_weight_selected_shallow5_1e-3.txt'
#cl.write_feature_weight2(weights,features,lambda1s,perfs[:,-1],uniqueness=False,tol=1e-3,filename=filename)
#filename='./result/GM12878_3Cl_feature_weight_unique_selected_shallow5_1e-2.txt'
#cl.write_feature_weight2(weights,features,lambda1s,perfs[:,-1],uniqueness=True,tol=1e-2,filename=filename)
#filename='./result/GM12878_3Cl_feature_weight_selected_shallow5_1e-2.txt'
#cl.write_feature_weight2(weights,features,lambda1s,perfs[:,-1],uniqueness=False,tol=1e-2,filename=filename)
gc_collect()
