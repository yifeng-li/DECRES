#!/usr/bin/env python
"""
An example of running multilayer perceptrons.
Yifeng Li
CMMT, UBC, Vancouver
Sep. 23, 2014
Contact: yifeng.li.cn@gmail.com
"""

#qsub -l procs=1,pmem=2000mb,walltime=12:00:00 -r n -N main_train_test_cv -o main_train_test_cv.out -e main_train_test_cv.err -M yifeng.li.cn@gmail.com -m bea main_rbm.py
import os
#os.environ['THEANO_FLAGS']='device=cpu,base_compile=/var/tmp'
import sys
import numpy

import rbm
import logistic_sgd
import classification as cl
from gc import collect as gc_collect

numpy.warnings.filterwarnings('ignore') # Theano causes some warnings  

# taking the input parameters
#cell=sys.argv[1] # cell type
#wid=sys.argv[2] # window size

path="/home/yifengli/prog/my/deep_learning_v1_1/"
os.chdir(path)

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

data_dir="/home/yifengli/prog/my/deep_learning_v1_1/data/"
result_dir="/home/yifengli/prog/my/deep_learning_v1_1/result/"

#cells=["GM12878","HepG2","K562","HelaS3","HUVEC","A549","MCF7","HMEC"]
#wids=[200,500,1000,2000,4000]
cells=["GM12878"]
wids=[200]

for cell in cells:
    for wid in wids:
        filename=data_dir + cell + "_" + str(wid) + "bp_Data.txt";
        data=numpy.loadtxt(filename,delimiter='\t',dtype='float32')
        filename=data_dir + cell + "_" + str(wid)  + "bp_Classes.txt";
        classes=numpy.loadtxt(filename,delimiter='\t',dtype=object)
        filename=data_dir+ cell + "_Features.txt"
        features=numpy.loadtxt(filename,delimiter='\t',dtype=object)
                    
        given=["A-E","I-E","A-P","I-P","A-X","I-X","UK"]
        #given=["A-E","I-E"]
        #given=["A-P","I-P"]
        #given=["A-E","A-P"]
        #given=["A-E","A-X"]
        #given=["A-P","A-X"]
        #given=["A-E","A-P","A-X"]
        #given=["A-E","I-E","A-P","I-P"]
        #given=["A-E","I-E","A-P","I-P","A-X","I-X"]
        #given=["I-E","I-P"]
        data,classes,_=cl.take_some_classes(data,classes,given=given,others=None)
        
        # balance the sample sizes of the classes
        rng=numpy.random.RandomState(1000)
        data,classes,others=cl.balance_sample_size(data,classes,others=None,min_size_given=None,rng=rng)

        print data.shape
        print numpy.unique(classes)

        #group=[["A-E"],["I-E"],["A-P"],["I-P"],["A-X"],["I-X"],["UK"]]
        #group=[["A-E","A-P"],["I-E","I-P","A-X","I-X","UK"]]
        #group=[["A-E","A-P","A-X"],["I-E","I-P","I-X","UK"]]
        group=[["A-E"],["A-P"],["I-E","I-P","A-X","I-X","UK"]]
        #group=[["A-E"],["A-P"],["A-X"],["I-E","I-P","I-X","UK"]]
        #group=[["A-E"],["I-E"]]
        #group=[["A-P"],["I-P"]]
        #group=[["A-E"],["A-P"]]
        #group=[["A-E"],["A-X"]]
        #group=[["A-P"],["A-X"]]
        #group=[["A-E"],["A-P"],["A-X"]]
        #group=[["A-E","I-E"],["A-P","I-P"]]
        #group=[["A-E","A-P"],["I-E","I-P"]]
        #group=[["A-E","I-E"],["A-P","I-P"],["A-X","I-X"]]
        #group=[["A-E","A-P","A-X"],["I-E","I-P","I-X"]]
        #group=[["I-E"],["I-P"]]
        classes=cl.merge_class_labels(classes,group)

        print numpy.unique(classes)

        classes_unique,classes=cl.change_class_labels(classes)
        
        print numpy.unique(classes)
        
        # set random state
        rng=numpy.random.RandomState(2000)
        data,classes,others=cl.balance_sample_size(data,classes,others=None,min_size_given=None,rng=rng)

        # permute data to speed up learning
        data_permute_id=rng.permutation(len(data))
        data=data[data_permute_id,:]
        classes=classes[data_permute_id]

        print data.shape
        print numpy.unique(classes)
        
        kfolds=10
        ind_folds=cl.kfold_cross_validation(classes,k=kfolds,shuffle=True,rng=rng)

        for i in range(kfolds):
            test_set_x_org=data[ind_folds==i,:]
            test_set_y_org=classes[ind_folds==i]
            train_set_x_org,train_set_y_org,valid_set_x_org,valid_set_y_org,_,_=cl.partition_train_valid_test(data[ind_folds!=i,:],classes[ind_folds!=i],ratio=(3,1,0),rng=rng)
            
            # normalization
            train_set_x_org,data_min,data_max=cl.normalize_col_scale01(train_set_x_org,tol=1e-10)
            valid_set_x_org,_,_=cl.normalize_col_scale01(valid_set_x_org,tol=1e-10,data_min=data_min,data_max=data_max)
            test_set_x_org,_,_=cl.normalize_col_scale01(test_set_x_org,tol=1e-10,data_min=data_min,data_max=data_max)

            # setting the parameter
            learning_rate=0.1
            alpha=0.1
            n_hidden=64
            persistent_chain_k=30
            n_epochs=100
            batch_size=100

            # train, and extract features from training set
            model_trained, train_set_x_extr, training_time = rbm.train_model(train_set_x_org=train_set_x_org, 
                                                                            training_epochs=n_epochs, batch_size=batch_size,
                                                                            n_hidden=n_hidden, learning_rate=learning_rate, persistent_chain_k=persistent_chain_k, rng=rng)
            
            # extract features from test set and validation set
            test_set_x_extr = rbm.test_model(model_trained, test_set_x_org)
            valid_set_x_extr =rbm.test_model(model_trained, valid_set_x_org)
    
            # classification
            # train classifier
            learning_rate=0.1
            n_epochs=100
            batch_size=100
            logistic_trained,training_time_logistic=logistic_sgd.train_model(learning_rate=learning_rate, n_epochs=n_epochs, 
                                                                             train_set_x_org=train_set_x_extr, train_set_y_org=train_set_y_org, 
                                                                             valid_set_x_org=valid_set_x_extr, valid_set_y_org=valid_set_y_org, 
                                                                             batch_size=batch_size)

            # test classifier       
            test_set_y_pred,test_set_y_pred_prob,test_time=logistic_sgd.test_model(logistic_trained,test_set_x_extr)
                        
            # evaluate classification performance
            perf_i,conf_mat_i=cl.perform(test_set_y_org,test_set_y_pred,numpy.unique(train_set_y_org))
            print perf_i
            print conf_mat_i
            if i==0:
                perf=perf_i
                conf_mat=conf_mat_i
                training_times=training_time
                test_times=test_time
            else:
                perf=numpy.vstack((perf,perf_i))
                conf_mat=conf_mat+conf_mat_i
                training_times=training_times + training_time
                test_times=test_times + test_time

        # calculate mean performance and std
        perf_mean=numpy.mean(perf,axis=0)
        perf_std=numpy.std(perf,axis=0)

        print perf_mean
        print perf_std
        print conf_mat
        # save the performance
        save_dir=result_dir + "_".join(classes_unique)
        try:
            os.makedirs(save_dir)
        except OSError:
            pass
        filename=cell + "_" + str(wid) + "bp.txt"
        cl.save_perform(save_dir,filename,perf=perf_mean,std=perf_std,conf_mat=conf_mat,classes_unique=classes_unique,training_time=training_times,test_time=test_times)
        gc_collect()
