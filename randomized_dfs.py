# randomized deep feature selection
from __future__ import division
import time
import numpy
import math
import deep_feat_select_mlp
import classification as cl


class randomized_dfs:
    def __init__(self,n_in,n_out):
        self.n_in=n_in
        self.num_features=n_in
        self.n_out=n_out
        self.num_classes=n_out

        
    def train(self,train_set_x_org=None, train_set_y_org=None,features=None,
              num_samplings=100,
              randomize_method="random_rescaling",
              random_rescaling_alpha=0.5,
              random_sampling_portion=0.66,
              learning_rate=0.1, alpha=0.01, 
              lambda1=0.001, lambda2=1.0, alpha1=0.001, alpha2=0.01, 
              n_hidden=[256,128,16], n_epochs=1000, batch_size=100, 
              activation_func="relu", rng=numpy.random.RandomState(100),
              dfs_select_method="top_num", dfs_threshold=0.001, dfs_top_num=10,
              max_num_epoch_change_learning_rate=80,max_num_epoch_change_rate=0.8,learning_rate_decay_rate=0.8):
        """
        Train the randomized DFS.
        num_samplings: int, number of reruns of DFS.
        randomize_method: string, the randomizing method, can be one of {"random_rescaling","random_sampling","random_sampling_and_random_rescaling"}.
        """
        if randomize_method=="random_rescaling":
            train_set_x,train_set_y,valid_set_x,valid_set_y,_,_=cl.partition_train_valid_test(train_set_x_org, train_set_y_org,ratio=(3,1,0), rng=rng)

        self.num_samplings=num_samplings
        self.randomize_method=randomize_method
        self.random_rescaling_alpha=random_rescaling_alpha
        self.random_sampling_portion=random_sampling_portion
        self.learning_rate=learning_rate
        self.alpha=alpha
        self.lambda1=lambda1
        self.lambda2=lambda2
        self.alpha1=alpha1
        self.alpha2=alpha2
        self.n_hidden=n_hidden
        self.n_epochs=n_epochs
        self.batch_size=batch_size
        self.activation_func="relu"
        self.max_num_epoch_change_learning_rate=max_num_epoch_change_learning_rate
        self.max_num_epoch_change_rate=max_num_epoch_change_rate
        self.learning_rate_decay_rate=learning_rate_decay_rate
        self.features=features
        self.classifiers=[]
        self.feature_counts=numpy.zeros(shape=(self.n_in,),dtype=int)
        self.feature_weights=numpy.zeros(shape=(self.n_in,self.num_samplings),dtype=float)
        self.training_time=0
        self.classes_unique=numpy.unique(train_set_y_org)
        self.rescale_Ws=[]
        for ns in range(self.num_samplings):
            print "The {0}-th run of randomized DFS...".format(ns)
            rng_ns=numpy.random.RandomState(ns)

            # generate a subsample of data points
            if randomize_method=="random_sampling" or randomize_method=="random_sampling_and_random_rescaling":
                # sample data
                train_set_x,train_set_y,ind_train,_=cl.sampling(train_set_x_org,train_set_y_org,others=None,portion=random_sampling_portion,max_size_given=None,rng=rng_ns)
                valid_set_x=numpy.delete(train_set_x_org,ind_train,axis=0)
                valid_set_y=numpy.delete(train_set_y_org,ind_train)
                # reorder the sampled data
                train_set_x,train_set_y,_=cl.sort_classes(train_set_x,train_set_y)
            if randomize_method=="random_rescaling" or randomize_method=="random_sampling_and_random_rescaling":
                rescale_ws=rng_ns.uniform(low=random_rescaling_alpha,high=1.0,size=(1,self.num_features))
                train_set_x=train_set_x*rescale_ws ### may multiplify, not division, leave it for future!!!!!!!!!!
                valid_set_x=valid_set_x*rescale_ws
                self.rescale_Ws.extend([rescale_ws])
                
            # run DFS
            classifier,training_time=deep_feat_select_mlp.train_model(train_set_x_org=train_set_x, train_set_y_org=train_set_y, 
                                                                      valid_set_x_org=valid_set_x, valid_set_y_org=valid_set_y, 
                                                                      learning_rate=learning_rate, alpha=alpha, lambda1=lambda1, lambda2=lambda2,
                                                                      alpha1=alpha1, alpha2=alpha2, n_hidden=n_hidden,
                                                                      n_epochs=n_epochs, batch_size=batch_size, activation_func=activation_func, rng=rng_ns,
                                                                      max_num_epoch_change_learning_rate=max_num_epoch_change_learning_rate,
                                                                      max_num_epoch_change_rate=max_num_epoch_change_rate,
                                                                      learning_rate_decay_rate=learning_rate_decay_rate)
            features_selected,logind_selected,weights_selected,weights=deep_feat_select_mlp.select_features(classifier,features,select_method=dfs_select_method,threshold=dfs_threshold,top_num=dfs_top_num)
            print weights
            self.classifiers.append(classifier)
            self.feature_counts=self.feature_counts+numpy.array(logind_selected,dtype=int)
            print self.feature_counts
            self.feature_weights[:,ns]=weights
            self.training_time=self.training_time+training_time

        # final clean up
        self.feature_importance=self.feature_counts/self.num_samplings
        print self.feature_importance
        return self.feature_importance,self.feature_weights,self.training_time


    def predict(self,test_set_x_org,batch_size=1000):
        start_time=time.clock()       
        num_test_samples=test_set_x_org.shape[0]
        #for each classifier, predict the test labels
        test_set_y_predicteds=numpy.zeros(shape=(num_test_samples,self.num_samplings),dtype=int)
        for ns in range(self.num_samplings):
            # rescale test samples as in training
            if self.randomize_method=="random_rescaling" or self.randomize_method=="random_sampling_and_random_rescaling":
                test_set_x=test_set_x_org*self.rescale_Ws[ns]
            # prediction
            classifier=self.classifiers[ns]
            test_set_y_predicted,_,_=deep_feat_select_mlp.test_model(classifier,test_set_x,batch_size=batch_size)
            print numpy.unique(test_set_y_predicted)
            test_set_y_predicteds[:,ns]=test_set_y_predicted
        # vote for class labels
        test_set_y_predicted,test_set_y_predicted_prob=self.committee_voting(test_set_y_predicteds,self.num_classes)
        end_time=time.clock()
        test_time=end_time-start_time
        return test_set_y_predicted,test_set_y_predicted_prob,test_time

    
    def committee_voting(self,test_set_y_predicteds,num_classes):
        """
        test_set_y_predicteds: numpy array, each column is the predictions in one run.
        """
        # ensemble results to probabilities
        num_test_samples=test_set_y_predicteds.shape[0]
        test_set_y_predicted_prob=numpy.zeros(shape=(num_test_samples,num_classes))
        for nt in range(num_test_samples):
            y_nt=test_set_y_predicteds[nt,:] # collectively predicted result of the nt-th test sample
            for c in range(num_classes):
                test_set_y_predicted_prob[nt,c]=numpy.sum(y_nt==c)

        # counts to prob
        test_set_y_predicted_prob=test_set_y_predicted_prob/self.num_samplings
        # prob to class labels
        test_set_y_predicted=numpy.argmax(test_set_y_predicted_prob,axis=1)
        
        print test_set_y_predicteds
        print test_set_y_predicted_prob
        print test_set_y_predicted
        
        return test_set_y_predicted,test_set_y_predicted_prob
            

    def save_feature_importance_weights(self,path="./",prefix="RDFS",ifsort=False,other=None):
        if ifsort:
            ind_fe=numpy.argsort(self.feature_importance,kind="mergesort")
            ind_fe=ind_fe[::-1]
            features=numpy.array(self.features)
            features=features[ind_fe]
            feature_importance=self.feature_importance[ind_fe]
            feature_weights=self.feature_weights[ind_fe,:]
            if other is not None:
                other=other[ind_fe]
        else:
            features=numpy.array(self.features)
            feature_importance=self.feature_importance
            feature_weights=self.feature_weights
            
        
        
        features.shape=(len(features),1)
        feature_importance=numpy.array(feature_importance,dtype=str)
        feature_importance.shape=(len(feature_importance),1)
        feature_weights=numpy.array(feature_weights,dtype=str)
        if other is None:
            feature_importance_weights=numpy.hstack((features,feature_importance,feature_weights))
        else:
            other=numpy.array(other,dtype=str)
            other.shape=(len(other),1)
            feature_importance_weights=numpy.hstack((features,feature_importance,other,feature_weights))
        if ifsort:
            filename=path + prefix + "_feature_importance_weights_sorted.txt"
        else:
            filename=path + prefix + "_feature_importance_weights.txt"
        numpy.savetxt(filename,feature_importance_weights,fmt="%s",delimiter="\t")
