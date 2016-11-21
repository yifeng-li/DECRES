"""
A module for generic classification purpose.

Funtionality include:

normalize_l2norm: Normalize each row has unit l_2 norm.

normalize_col_scale01: Normalize each feature (column) to scale [0,1].

normalize_row_scale01: Normalize each sample (row) to scale [0,1].

normalize_mean0std1: Normalize each feature to have mean 0 and std 1.

normalize_matrical_samples: Normalize matrical samples.

balance_sample_size: Balance sample size of a data set among classes.  

change_class_labels: Change class labels to {0,1,2,3,...,C-1}.

change_class_labels_back: Change class labels from {0,1,2,..,C-1} to C given labels.

change_class_labels_to_given: Change original class labels to a given labels.

merge_class_labels: Merge class labels into several super groups/classes.

take_some_classes: Only take sevaral classes, and remove the rest.

partition_train_valid_test: Partition the whole data into training, validation, and test sets.

kfold_cross_validation: k-fold cross-validation.

reduce_sample_size: Reduce sample by to 1/times.

take_unique_features: Take unqiue features and make the change in the data accordingly.

take_unique_features_large: Take unqiue features and make the change in a big data accordingly. Write the resulted data into a txt file.

take_common_features: Return common features and their indices.

perform: Compute the classification performance given predicted and actual class labels.

con_mat_to_num: Compute performance given confusion matrix.

save_perform: Save performance to a txt file.

write_feature_weight: Write the weights of the input layer of a DFS model to a file. Only applicable to deep feature selection.

write_feature_weight2: Write the weights of the input layer of a DFS and other information (accuracy, feature subsets) to a file.  Only applicable to deep feature selection.

plot_bar_group:  Plot grouped bars given a matrix.

plot_3dbar_group: Plot grouped 3d-bars given a matrix.

plot_bar_group_subplots: Plot subplots of the (classification) performance.

plot_box_multi:  Plot multiple boxes in a plot according to class information.

Yifeng Li
CMMT, UBC, Vancouver
Sep 23, 2014
Contact: yifeng.li.cn@gmail.com
"""
from __future__ import division
import numpy as np
#from sklearn import cross_validation
import math
import os
import sys


def normalize_l2norm(data,tol=0):
    """
    Normalize each row has unit l_2 norm. 
    
    INPUTS:
    data: numpy 2d array or matrix, each row should be a sample.
    tol: tolerance to avoid errors.
    
    OUTPUTS:
    data: numpy 2d array or matrix, normalized data.
    
    Example:
    data=[[3,5,7,9],[3.0,2,1.1,8.4],[5.9,9,8,10]]
    data=np.array(data)
    data_normalized=normalize_l2norm(data)
    print data_normalized
    """
    data_sqrt=np.sqrt(np.square(data).sum(axis=1))
    data_sqrt.shape=(data_sqrt.shape[0],1)
    #tol=0#1e-8
    data=data/(data_sqrt+tol)
    return data


def normalize_col_scale01(data,tol=1e-6,data_min=None,data_max=None,clip=False,clip_min=1e-3,clip_max=1e3):
    """
    Normalize each feature (column) to scale [0,1]. 
    
    INPUTS:
    data: numpy 2d array or matrix, each row should be a sample.

    tol: tolerance to avoid errors.

    data_min: numpy 1d array or vector, the minmum values of the columns.

    data_max: numpy 1d array or vector, the maxmum values of the columns.
    
    OUTPUTS:
    data: numpy 2d array or matrix, normalized data.
    
    data_min: numpy 1d array or vector, the minmum values of the columns.

    data_max: numpy 1d array or vector, the maxmum values of the columns.
    
    Example: ...
    """
    if clip:
        data[data<clip_min]=clip_min
        data[data>clip_max]=clip_max
    if data_max is None:
        data_max=np.max(data,axis=0)
        data_max.reshape((1,data_max.shape[0]))
    if data_min is None:
        data_min=np.min(data,axis=0)
        data_min.reshape((1,data_min.shape[0]))
    #tol=0#1e-8
    return (data-data_min)/(data_max-data_min+tol),data_min,data_max


def normalize_row_scale01(data,tol=1e-6,data_min=None,data_max=None,clip=False,clip_min=1e-3,clip_max=1e3):
    """
    Normalize each sample (row) to scale [0,1]. 
    
    INPUTS:
    data: numpy 2d array or matrix, each row should be a sample.

    tol: tolerance to avoid errors.

    data_min: numpy 1d array or vector, the minmum values of the rows.

    data_max: numpy 1d array or vector, the maxmum values of the rows.

    OUTPUTS:
    data: numpy 2d array or matrix, normalized data.

    data_min: numpy 1d array or vector, the minmum values of the rows.

    data_max: numpy 1d array or vector, the maxmum values of the rows.
    
    Example: ...
    """
    if clip:
        data[data<clip_min]=clip_min
        data[data>clip_max]=clip_max

    if data_max is None:
        data_max=np.max(data,axis=1)
        data_max.shape=(data_max.shape[0],1)
    #if clip:
    #	data_max[data_max>clip_max]=clip_max
    if data_min is None:
        data_min=np.min(data,axis=1)
        data_min.shape=(data_min.shape[0],1)
    #if clip:
    #    data_min[data_min<clip_min]=clip_min
    #tol=1e-6#1e-8
    return (data-data_min)/(data_max-data_min+tol),data_min,data_max


def normalize_mean0std1(data,data_mean=None,data_std=None,tol=1e-6):
    """
    Normalize each feature (feature) to mean 0 and std 1.
    
    INPUTS:
    data: numpy 2d array or matrix, each row should be a sample.
    
    data_mean: numpy 1d array or vector, the given means of samples, useful for normalize test data.
    
    data_std: numpy 1d array or vector, the given standard deviation of samples, useful for normalize test data.

    tol: tolerance to avoid errors.
    
    OUTPUTS:
    data: numpy 2d array or matrix, normalized data.
    
    data_mean: numpy 1d array or vector, the given means of samples, useful for normalize test data.
    
    data_std: numpy 1d array or vector, the given standard deviation of samples, useful for normalize test data.
    """
    if data_mean is None:
        data_mean=np.mean(data,axis=0)
    data_mean.reshape((1,data_mean.shape[0]))
    if data_std is None:
        data_std=np.std(data,axis=0)
    data_std.reshape((1,data_std.shape[0]))
    #tol=0#1e-8
    return (data-data_mean)/(data_std+tol),data_mean,data_std


def normalize_deseq(data,size_factors=None):
    # DESEQ normalization
    if size_factors is not None:
        data_normalized=data/size_factors
        return data_normalized,size_factors

    num_features,num_samples=data.shape
    geometric_means=np.zeros(shape=(num_features,1),dtype=float)
    for f in range(num_features):
        if np.any(data[f,:]==0):
            geometric_means[f,0]=0
        else:
            geometric_means[f,0]=2**(np.mean(np.log2(data[f,:])))
    #geometric_means=self.data_noheader_nofeature.prod(axis=1)**(1.0/self.num_samples) # geometric mean across all samples
    print geometric_means
    print np.median(geometric_means)
    print geometric_means.shape

    #geometric_means.shape=(len(geometric_means),1)
    #data_div_means=data_noheader_nofeature_copy/geometric_means
    #data_div_means=numpy.ma.masked_where(numpy.logical_or(data_div_means==numpy.inf,data_div_means==numpy.nan), data_div_means)
    #print data_div_means[1:10,:]
    #self.size_factors=numpy.ma.median(data_div_means, axis=0).filled(0) # masked median

    # compute size factors
    ind_geometric_means=geometric_means[:,0]!=0
    print ind_geometric_means.shape
    print "Total number of geometric means not equal to zero:{}".format(ind_geometric_means.sum())
    size_factors=np.zeros(shape=(1,num_samples),dtype=float)
    for s in range(num_samples):
        size_factors[0,s]=np.median(data[ind_geometric_means,s]/(geometric_means[ind_geometric_means,0]))

    if np.any(size_factors==0):
        print "Warning: at least one size factor = 0!"
    data_normalized=data/size_factors
    return data_normalized,size_factors

 
def normalize_matrical_samples(data,num_signal,method="l2norm"):
    """ 
    Normalize matrical samples.
    
    INPUTS:
    data: numpy 2d array or matrix, each row is a vectorized sample.

    num_signal: scalar, number of features in each sample (this parameter is needed to convert a vectorized sample into a matrix.).

    method: string, method to normalize each feature (or signal), can be "l2norm", "mean0std1", "scale01".

    OUTPUTS:
    change the data in-place.
    
    """
    feat_total=data.shape[1]
    feat_each=feat_total//num_signal
    for i in range(num_signal):
        if method=="l2norm":
            data[:,i*feat_each:(i+1)*feat_each]=normalize_l2norm(data[:,i*feat_each:(i+1)*feat_each])
        if method=="mean0std1":
            data[:,i*feat_each:(i+1)*feat_each],data_mean,data_std=normalize_mean0std1(data[:,i*feat_each:(i+1)*feat_each])
        if method=="scale01":
            data[:,i*feat_each:(i+1)*feat_each],data_min,data_max=normalize_row_scale01(data[:,i*feat_each:(i+1)*feat_each],tol=1e-6)


def balance_sample_size(data,classes,others=None,min_size_given=None,rng=np.random.RandomState(100)):
    """
    Balance sample size of a data set among classes by reducing the large classes.
    
    INPUTS:
    data: numpy 2d array or matrix, each row should be a sample.
    
    classes: numpy 1d array or vector, class labels.
    
    others: numpy 2d array or matrix, extra information of samples if available,
    each row should associated to a row of data.
    
    min_size_given: int, the size of each class wanted.
    
    rng: numpy random state.
    
    OUTPUTS:
    data: numpy 2d array or matrix, each row should be a sample, balanced data.
    
    classes: numpy 1d array or vector, balanced class labels.
    
    others: numpy 2d array or matrix, balanced other information.
    
    Example:
    data=[[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7]]
    data=np.array(data)
    classes=np.array(['zz','xx','xx','yy','zz','yy','xx'])
    balance_sample_size(data,classes)
    """    
    u, indices = np.unique(classes,return_inverse=True)
    indices=np.asarray(indices)
    num_u=len(u)
    sample_sizes=[]
    
    # get sample size of each class
    for i in xrange(num_u):
        sample_size_this=np.sum(indices==i)
        sample_sizes.append(sample_size_this)     
        
    size_min=np.amin(sample_sizes) # smallest sample size
    
    if min_size_given and size_min>min_size_given:
        size_min=min_size_given   
        
    indices_all=np.array([],dtype=indices.dtype)
    indices_range=np.array(range(len(indices)))
    
    for i in xrange(num_u):
        ind_this_num=indices_range[indices==i]
        ind_this_reduced=ind_this_num[rng.choice(len(ind_this_num),size=size_min,replace=False)]
        indices_all=np.append(indices_all,ind_this_reduced)
    
    # reduce the data    
    data=data[indices_all]
    classes=classes[indices_all]
    if np.any(others):
        others=others[indices_all]
    return data,classes,others


def balance_sample_size_increase(data,classes,others=None,max_size_given=None,rng=np.random.RandomState(100)):
    """
    Balance sample size of a data set among classes by increasing the sample size of small classes.
    
    INPUTS:
    data: numpy 2d array or matrix, each row should be a sample.
    
    classes: numpy 1d array or vector, class labels.
    
    others: numpy 2d array or matrix, extra information of samples if available,
    each row should associated to a row of data.
    
    max_size_given: int, the size of each class wanted.
    
    rng: numpy random state.
    
    OUTPUTS:
    data: numpy 2d array or matrix, each row should be a sample, balanced data.
    
    classes: numpy 1d array or vector, balanced class labels.
    
    others: numpy 2d array or matrix, balanced other information.
    
    Example:
    data=[[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7]]
    data=np.array(data)
    classes=np.array(['zz','xx','xx','yy','zz','yy','xx'])
    balance_sample_size_increase(data,classes)
    """    
    u, indices = np.unique(classes,return_inverse=True)
    indices=np.asarray(indices)
    num_u=len(u)
    sample_sizes=[]
    
    # get sample size of each class
    for i in xrange(num_u):
        sample_size_this=np.sum(indices==i)
        sample_sizes.append(sample_size_this)     
        
    size_max=np.amax(sample_sizes) # largest sample size
    
    if max_size_given and size_max<max_size_given:
        size_max=max_size_given   
        
    indices_all=np.array([],dtype=indices.dtype)
    indices_range=np.array(range(len(indices)))
    
    for i in xrange(num_u):
        ind_this_num=indices_range[indices==i]
        #replacetf=True if sample_sizes[i]<size_max else False
        if sample_sizes[i]>=size_max:
            ind_this_increased=ind_this_num[rng.choice(sample_sizes[i],size=size_max,replace=False)]
            indices_all=np.append(indices_all,ind_this_increased)
        else: # make sure each sample is used at least once
            ind_this_increased=ind_this_num
            ind_this_increased2=ind_this_num[rng.choice(sample_sizes[i],size=size_max-sample_sizes[i],replace=True)]
            indices_all=np.append(indices_all,ind_this_increased)
            indices_all=np.append(indices_all,ind_this_increased2)
    
    # increase the data    
    data=data[indices_all]
    classes=classes[indices_all]
    if np.any(others):
        others=others[indices_all]
    return data,classes,others


def summarize_classes(classes):
    """
    Print a summary of the classes.
    """
    u, indices = np.unique(classes,return_inverse=True)
    num_u=len(u)
    print "****************************"
    print "Number of samples: {0}".format(len(classes))
    print "Number of Classes:{0}".format(num_u)
    for c in u:
        num_c=np.sum(classes==c)
        print "Class {0}: {1} Samples".format(c,num_c)
    print "****************************"


def sort_classes(data,classes,others=None):
    """
    Group the class labels into blocks, if the class lables distribue randomly in the vector.
    data: numpy array, each row is a sample.
    classes: numpy 1d array, the class labels.
    """
    indices = np.argsort(classes,kind="mergesort")
    #print indices
    data=data[indices,:]
    classes=classes[indices]
    if others is not None:
        others=others[indices]
    return data,classes,others


def truncate_sample_size(data,classes,others=None,max_size_given=None,rng=np.random.RandomState(100)):
    """
    Balance sample size of a data set among classes.
    
    INPUTS:
    data: numpy 2d array or matrix, each row should be a sample.
    
    classes: numpy 1d array or vector, class labels.
    
    others: numpy 2d array or matrix, extra information of samples if available,
    each row should associated to a row of data.
    
    min_size_given: int, the size of each class wanted.
    
    rng: numpy random state.
    
    OUTPUTS:
    data: numpy 2d array or matrix, each row should be a sample, balanced data.
    
    classes: numpy 1d array or vector, balanced class labels.
    
    indices_all: numpy 1d array, the numerical indices of samples selected.

    others: numpy 2d array or matrix, balanced other information.
    
    Example:
    data=[[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7]]
    data=np.array(data)
    classes=np.array(['zz','xx','xx','yy','zz','yy','xx'])
    balance_sample_size(data,classes,others=NOne,max_size_given=50)
    """    
    u, indices = np.unique(classes,return_inverse=True)
    indices=np.asarray(indices)
    num_u=len(u)
    sample_sizes=[]
    
    # get sample size of each class
    for i in xrange(num_u):
        sample_size_this=np.sum(indices==i)
        sample_sizes.append(sample_size_this)
    sample_sizes=np.array(sample_sizes,dtype=int)
        
    size_min=np.amin(sample_sizes) # smallest sample size
    size_max=np.amax(sample_sizes) # largest sample size
    
    if size_max<max_size_given:
    	max_size_given=size_max
    sample_sizes[sample_sizes>max_size_given]=max_size_given        

    indices_all=np.array([],dtype=indices.dtype)
    indices_range=np.array(range(len(indices)))
    
    for i in xrange(num_u):
        ind_this_num=indices_range[indices==i]
        ind_this_reduced=ind_this_num[rng.choice(len(ind_this_num),size=sample_sizes[i],replace=False)]
        indices_all=np.append(indices_all,ind_this_reduced)
    
    # reduce the data    
    data=data[indices_all,:]
    classes=classes[indices_all]
    if np.any(others):
        others=others[indices_all]
    return data,classes,indices_all,others
    
    
def sampling(data,classes,others=None,portion=0.9,max_size_given=None,rng=np.random.RandomState(100)):
    """
    Sample data points for a given portion and upper limit.
    
    INPUTS:
    data: numpy 2d array or matrix, each row should be a sample.
    
    classes: numpy 1d array or vector, class labels.
    
    others: numpy 2d array or matrix, extra information of samples if available,
    each row should associated to a row of data.

    portion: float, portion of data points to be sampled.
    
    max_size_given: int, upper limit of the sampled data points in each class; if None, no limit.
    
    rng: numpy random state.
    
    OUTPUTS:
    data: numpy 2d array or matrix, each row should be a sample, balanced data.
    
    classes: numpy 1d array or vector, balanced class labels.
    
    indices_all: numpy 1d array, the numerical indices of samples selected.

    others: numpy 2d array or matrix, balanced other information.
    
    Example:
    data=[[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7]]
    data=np.array(data)
    classes=np.array(['zz','xx','xx','yy','zz','yy','xx'])
    sampling(data,classes,others=None,portion=0.6,max_size_given=50)
    """    
    u, indices = np.unique(classes,return_inverse=True)
    indices=np.asarray(indices)
    num_u=len(u)
    sample_sizes=[]
    
    # get sample size of each class
    for i in xrange(num_u):
        sample_size_this=np.sum(indices==i)
        sample_sizes.append(sample_size_this)
    sample_sizes=np.array(sample_sizes,dtype=int)
    sample_sizes=sample_sizes*portion
    sample_sizes=np.array(sample_sizes,dtype=int)
    # set a ceiling/limit
    if max_size_given is not None:
        sample_sizes[sample_sizes>max_size_given]=max_size_given  

    indices_all=np.array([],dtype=indices.dtype)
    indices_range=np.array(range(len(indices)))

    # sampling
    for i in xrange(num_u):
        ind_this_num=indices_range[indices==i]
        ind_this_reduced=ind_this_num[rng.choice(len(ind_this_num),size=sample_sizes[i],replace=False)]
        indices_all=np.append(indices_all,ind_this_reduced)
    
    # reduce the data    
    data=data[indices_all,:]
    classes=classes[indices_all]
    if np.any(others):
        others=others[indices_all]
    return data,classes,indices_all,others
    

def sampling_class_portion(data,classes,others=None,class_portion=None,rng=np.random.RandomState(100)):
    """
    Sampling data points in each class to keep a given portion among classes.
    class_portion: dict, the portion for each class, each value should be at least 1, e.g. class_portion={"class0":5,"class1":1,"class3":2}
    """
    u, indices = np.unique(classes,return_inverse=True)
    indices=np.asarray(indices)
    num_u=len(u)
    sample_sizes=dict()
    
    # get sample size of each class
    size_min=float("inf")
    for i in xrange(num_u):
        sample_size_this=np.sum(indices==i)
        sample_sizes[u[i]]=sample_size_this
        if class_portion[u[i]]==1 and sample_size_this<size_min:
            size_min=sample_size_this
    print size_min

    indices_all=np.array([],dtype=indices.dtype)
    indices_range=np.array(range(len(indices)))
    
    # sampling
    for i in xrange(num_u):
        ind_this_num=indices_range[indices==i]
        replacetf=True if sample_sizes[u[i]]<(size_min*class_portion[u[i]]) else False
        ind_this_reduced=ind_this_num[rng.choice(sample_sizes[u[i]],size=size_min*class_portion[u[i]],replace=replacetf)]
        indices_all=np.append(indices_all,ind_this_reduced)
    
    # get the sampled data    
    data=data[indices_all,:]
    classes=classes[indices_all]
    if np.any(others):
        others=others[indices_all]
    return data,classes,indices_all,others
    
    
def change_class_labels(classes):
    """
    Change class labels to {0,1,2,3,...,C-1}.
    
    INPUTS:
    classes: numpy 1d array or vector, the original class labels.
    
    OUTPUTS:
    u: numpy 1d array or vector, the unique class labels of the original class labels.
    
    indices: numpy 1d array or vector, the new class labels from {0,1,2,3,...,C-1}.
    
    Example:
    classes=['c2','c3','c2','c1','c2','c1','c3','c2']
    change_class_labels(classes)
    Yifeng Li, in UBC
    Aug 22, 2014.
    """
    u,indices=np.unique(classes,return_inverse=True)
    return u,indices   


def change_class_labels_back(classes,given):
    """
    Change class labels from {0,1,2,..,C-1} to C given labels.
    
    INPUTS:
    classes: numpy 1 d array or vector, the original class labels.
    
    given: list of new labels.
    
    OUTPUTS:
    classes_new: numpy 1 d array or vector, changed class labels.
    
    Example:
    classes=[1,2,0,0,2,1,1,2]
    given=["class0","class1","class2"]
    change_class_labels_to_given(classes,given)
    """
    classes=np.asarray(classes)
    classes_new=np.zeros(classes.shape,dtype=object)
    for i in range(len(given)):
        classes_new[classes==i]=given[i]
    return classes_new


def change_class_labels_to_given(classes,given):
    """
    Change original class labels to given labels.
    
    INPUTS:
    classes: numpy 1 d array or vector, the original class labels.
    
    given: dic, pairs of old and new labels. Or list of new labels.
    
    OUTPUTS:
    classes_new: numpy 1 d array or vector, changed class labels.
    
    Example:
    classes=[1,2,0,0,2,1,1,2]
    given={1:"class1", 2:"class2", 0:"class0"}
    # given=["class0","class1","class2"]
    change_class_labels_to_given(classes,given)
    """
    classes=np.asarray(classes)
    classes_new=np.zeros(classes.shape,dtype=object)
    for i in given:
        classes_new[classes==i]=given[i]
    return classes_new

    
def membership_vector_to_indicator_matrix(z,z_unique=None):
    """
    Extend membership vector z to binary indicator matrix Z.
    z: list or numpy vector, numerical class labels of samples.
    z_unique: list or numpy vector, the unique class labels, useful for tranforming class labels of test samples.
    For example:
    z=[-1,-1,-1,0,0,0,1,1,2,2,2]
    Z=[[1,0,0,0],
       [1,0,0,0],
       [1,0,0,0],
       [0,1,0,0],
       [0,1,0,0],
       [0,1,0,0],
       [0,0,1,0],
       [0,0,1,0],
       [0,0,0,1],
       [0,0,0,1],
       [0,0,0,1]]
    """
    z=np.array(z,dtype=int)
    if z_unique is None:
        z_unique=np.unique(z)
    M=len(z)
    U=len(z_unique)
    Z=np.zeros(shape=(M,U),dtype=int)
    for m in range(M):
        for u in range(U):
            if z[m]==z_unique[u]:
                Z[m,u]=1
    return Z,z_unique


def merge_class_labels(classes,group):
    """
    Merge class labels into several super groups/classes.
    
    INPUTS:
    classes: numpy 1 d array or vector, the original class labels.
    
    group: tuple of tuples or lists, 
    group[i] indicates which original classes to be merged to the i-th super class.
    
    OUTPUTS:
    classes_merged: numpy 1 d array or vector, the merged class labels.
    If original labels are strings, they are concatenated by "+".
    If original lables are numbers, they are renumbered starting from 0.
    
    Example
    classes=[0,3,4,2,1,3,3,2,4,1,1,0,0,1,2,3,4,1]
    group=([0],[1,2],[3,4])
    merge_class_labels(classes,group)
    classes=['c2','c1','c0','c0','c1','c2','c1']
    group=(['c0'],['c1','c2'])
    merge_class_labels(classes,group)
    """
    classes=np.asarray(classes)
    if (classes.dtype != int) and (classes.dtype != 'int64') and (classes.dtype != 'int32'):
        classes_merged=np.zeros(classes.shape,dtype=object)
        for subgroup in group:
            subgroup_label='+'.join(subgroup)
            for member in subgroup:
                classes_merged[classes==member]=subgroup_label
    else: # int class labels
        classes_merged=np.zeros(classes.shape,dtype=int)
        for i in range(len(group)):
            subgroup=group[i]
            for member in subgroup:
                classes_merged[classes==member]=i
    
    return classes_merged
 
    
def take_some_classes(data,classes,given,others=None):
    """
    Only take sevaral classes, and remove the rest.
    
    INPUTS:
    data: numpy 2d array or matrix, each row is a sample, the original data.
    
    classes: numpy 1d array or vector, class labels, the original labels.
    
    given: numpy 1d array or vector, indicates which classes to be taken.

    others: numpy 1d or 2d array (vector or matrix), others related data, e.g. regions corresponding to the classes.
        
    OUTPUTS:
    data: numpy 2d array or matrix, each row is a sample, the taken data.
    
    classes: numpy 1d array or vector, class labels, the taken labels.

    others: numpy 1d or 2d array (vector or matrix), the taken "others".
    """
    classes=np.asarray(classes)
    log_ind=np.zeros(classes.shape,dtype=bool)
    for i in range(len(given)):
        log_ind[classes==given[i]]=True
    classes=classes[log_ind]
    data=data[log_ind]
    if np.any(others):
        others=others[log_ind]
    return data,classes,others


def partition_train_valid_test(data, classes,ratio=(1,1,1), rng=np.random.RandomState(1000)):
    """
    Partition the whole data into training, validation, and test sets.
    
    INPUTS:
    data: numpy 2d array or matrix, each row is a sample, the original data.
    
    classes: numpy 1d array or vector, class labels, the original labels.
    
    ratio, int tuple or list of length 3, (ratio_of_train_set,ratio_of_valid_set,ratio_test_set).
    
    random_state: random state to generate the random numbers, can be e.g. random_state=1000, default is None.

    OUTPUTS:
    train_set_x: data of training set.
    
    train_set_y: class labels of training set.
    
    valid_set_x: data of validation set.
    
    valid_set_y: class labels of validation set.
    
    test_set_x: data of test set.
    
    test_set_y: class labels of test set.
    
    Example:
    data=np.random.random((20,3))
    classes=np.array([0,2,2,2,0,0,1,1,0,0,0,2,2,2,0,0,1,1,0,0],dtype=int)
    train_set_x,train_set_y,valid_set_x,valid_set_y,test_set_x,test_set_y \
    =partition_train_valid_test(data,classes,ratio=(2,1,1))
    Yifeng Li, in UBC.
    August 22, 2014.    
    """
    k=sum(ratio) # ratio must be a vector of integers
    ind=kfold_cross_validation(classes,k=k,shuffle=True,rng=rng)
    sequence=np.arange(len(classes))
    train_ind=np.array([],dtype=int)
    valid_ind=np.array([],dtype=int)
    test_ind=np.array([],dtype=int)
    count=0
    for ki in range(k):
        if count<ratio[0]:
            train_ind=np.append(train_ind,sequence[ind==ki])
            count=count+1
            continue
        if count>=ratio[0] and count <ratio[0]+ratio[1]:
            valid_ind=np.append(valid_ind,sequence[ind==ki])
            count=count+1
            continue
        if count>=ratio[0]+ratio[1] and ratio[2]>0:
            test_ind=np.append(test_ind,sequence[ind==ki])
            count=count+1
            continue
    train_set_x=data[train_ind]
    train_set_y=classes[train_ind]
    valid_set_x=data[valid_ind]
    valid_set_y=classes[valid_ind]
    test_set_x=data[test_ind]
    test_set_y=classes[test_ind]
    return train_set_x,train_set_y,valid_set_x,valid_set_y,test_set_x,test_set_y


def partition_train_valid_test2(data, classes, others, ratio=(1,1,1), rng=np.random.RandomState(1000)):
    """
    Partition the whole data into training, validation, and test sets. The only difference between partition_train_valid_test2 and partition_train_valid_test is that the former can handle other information.
    
    INPUTS:
    data: numpy 2d array or matrix, each row is a sample, the original data.
    
    classes: numpy 1d array or vector, class labels, the original labels.

    others: numpy 2d array or matrix, extra information of samples if available,
    each row should associated to a row of data.
    
    ratio, int tuple or list of length 3, (ratio_of_train_set,ratio_of_valid_set,ratio_test_set).
    
    OUTPUTS:
    train_set_x: data of training set.
    
    train_set_y: class labels of training set.

    train_set_others.
    
    valid_set_x: data of validation set.
    
    valid_set_y: class labels of validation set.
    
    valid_set_others.

    test_set_x: data of test set.
    
    test_set_y: class labels of test set.

    test_set_others.
    
    Yifeng Li, in UBC.
    August 22, 2014.    
    """
    k=sum(ratio) # ratio must be a vector of integers
    ind=kfold_cross_validation(classes,k=k,shuffle=True,rng=rng)
    sequence=np.arange(len(classes))
    train_ind=np.array([],dtype=int)
    valid_ind=np.array([],dtype=int)
    test_ind=np.array([],dtype=int)
    count=0
    for ki in range(k):
        if count<ratio[0]:
            train_ind=np.append(train_ind,sequence[ind==ki])
            count=count+1
            continue
        if count>=ratio[0] and count <ratio[0]+ratio[1]:
            valid_ind=np.append(valid_ind,sequence[ind==ki])
            count=count+1
            continue
        if count>=ratio[0]+ratio[1] and ratio[2]>0:
            test_ind=np.append(test_ind,sequence[ind==ki])
            count=count+1
            continue
    train_set_x=data[train_ind]
    train_set_y=classes[train_ind]
    if others is not None:
	train_set_others=others[train_ind]
    else:
	train_set_others=None
    valid_set_x=data[valid_ind]
    valid_set_y=classes[valid_ind]
    if others is not None:
	valid_set_others=others[valid_ind]
    else:
	valid_set_others=None
    test_set_x=data[test_ind]
    test_set_y=classes[test_ind]
    if others is not None:
	test_set_others=others[test_ind]
    else:
	test_set_others=None
	
    return train_set_x,train_set_y,train_set_others,valid_set_x,valid_set_y,valid_set_others,test_set_x,test_set_y,test_set_others


def kfold_cross_validation(classes,k,shuffle=True,rng=np.random.RandomState(1000)):
    """
    kfold cross-validation.

    INPUTS: 
    classes: numpy 1d array of vector.
    
    k: scalar, the number of folds.

    shuffle: logical, whether need to shuffle the distribution of each fold.

    rng: random number generator.

    OUTPUTS:
    indices_folds: numpy 1d array, the splits. For example if k=3, indices_folds can be [2,2,1,0,1,2,0,1,1,2,0,0].

    Yifeng Li, April 02, 2015, in UBC.
    """
    num_samples=len(classes)
    classes=np.array(classes)
    classes_unique=np.unique(classes)
    num_classes=len(classes_unique)
    indices=np.arange(num_samples)
    indices_folds=np.zeros([num_samples],dtype=int)
    for cl in classes_unique:
        indices_cl=indices[classes==cl]
        num_samples_cl=len(indices_cl)

        # split this class into k parts
        if shuffle:
            rng.shuffle(indices_cl) # in-place shuffle
        
        # module and residual
        num_samples_each_split=num_samples_cl//k
        res=num_samples_cl%k
        num_samples_splits=num_samples_each_split*np.ones([k],dtype=int)
        if res>0:
            for r in np.arange(res):
                num_samples_splits[r]=num_samples_splits[r]+1
        
        # for each part, assign 0,1,2,...,k-1
        start=0
        end=0
        
        for ki in range(k):
            start=end
            end=end+num_samples_splits[ki]
            indices_folds[indices_cl[start:end]]=ki
            
    return indices_folds


def factor_sizes_to_factor_labels(z,start=-1):
    # z is a tuple e.g. (3,2,3) of a list e.g. [3,2,3]
    labels=[]
    for i in z:
        labels.extend([start]*i)
        start=start+1
    #print labels
    return labels


def perform(y,y_predicted,unique_classes):
    """
    Compute the classification performance given predicted and actual class labels.
    
    INPUTS: 
    y: numpy 1d array or vector, the actual class labels.
    
    y_predicted: numpy 1d array or vector, the predicted class labels.
    
    unique_classes: numpy 1d array or vector of length C (# classes), all unique actual class labels.
    
    OUTPUTS:
    perf: numpy 1d array or vector of length 2*C+3, 
    [acc_0, acc_1, acc_{C-1}, accuracy, balanced accuracy].
    For two-classes, perf=[sensitivity, specificity, PPV, NPV, accuracy, averaged sensitivity (or called balanced accuracy, that is 0.5*(sen+spec) ), averaged PVs (that is 0.5*(PPV+NPV) )].
    For multi-classes, perf=[sen_0, sen_1, ..., sen_{C-1}, precision_0, precision_1, ..., precision_{C-1}, accuracy, averaged sensitivity, averaged percision] that is [class-wise rates, class-wise predictive rates, accuracy, averaged class-wise rate, averaged class-wise predictive rate]
    
    confusion_matrix: numpy 2d array of size C X C, confusion matrix.
    
    Example:
    y=[0,0,1,1,1,2,2,2,2]
    y_predicted=[0,1,1,1,2,2,2,0,1]
    perform(y,y_predicted,[0,1,2])
    Yifeng Li, in UBC.
    August 23, 2014.
    """
    y=np.asarray(y,dtype=int)
    y_predicted=np.asarray(y_predicted,dtype=int)
    
    numcl=len(unique_classes)
    confusion_matrix=np.zeros((numcl,numcl),dtype=float)
    for i in xrange(len(y)):
        confusion_matrix[y[i],y_predicted[i]]=confusion_matrix[y[i],y_predicted[i]]+1
    perf=np.zeros((2*numcl+3,)) # sensitivity_0,sensitivity_1,...,sensitivity_{C-1}, precision_0,precision_1,...,precision_{C-1}, accuracy, balanced sensitivity, balanced precision 
    perf[0:numcl]=confusion_matrix.diagonal()/confusion_matrix.sum(axis=1) # sensitivity and specifity for two classes, (class-wise rates for multi-classes)
    perf[numcl:2*numcl]=confusion_matrix.diagonal()/confusion_matrix.sum(axis=0) # PPV and NPV for two classes, (class-wise predictive rates for multi-classes)
    perf[2*numcl]=confusion_matrix.diagonal().sum()/confusion_matrix.sum(axis=1).sum() # accuracy
    perf[2*numcl+1]=np.mean(perf[0:numcl]) # balanced accuracy for two-classes, average class-wise rate for multi-class 
    perf[2*numcl+2]=np.mean(perf[numcl:2*numcl]) # avarage class-wise predictive rate
    return perf,confusion_matrix


def prc(test_set_y_org,test_set_y_pred_prob,classes_unique,plot_curve=False,filename="./fig_prc.pdf",colors=None,positive_class_for_two_classes=None,figwidth=5,figheight=5):
    """
    Calcuate the area under precision-recall curve, and draw the precision-recall curve.
    """
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from scipy import interp
    import matplotlib as mpl
    mpl.use("pdf")
    import matplotlib.pyplot as plt
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes=len(classes_unique)
    test_set_Y_org,test_set_y_org_unique=membership_vector_to_indicator_matrix(test_set_y_org)
    
    for c in range(n_classes):
        precision[c], recall[c], _ = precision_recall_curve(test_set_Y_org[:, c], test_set_y_pred_prob[:, c])
        average_precision[c] = average_precision_score(test_set_Y_org[:, c], test_set_y_pred_prob[:, c])
        
    # Compute macro-average ROC curve and AUROC area
    # First aggregate all recalls
    all_recall = np.unique(np.concatenate([recall[c] for c in range(n_classes)]))
    #all_recall = np.sort(np.concatenate([recall[c] for c in range(n_classes)]))
    # Then interpolate all PRC curves at this points
    mean_precision = np.zeros_like(all_recall)
    for c in range(n_classes):
        mean_precision = mean_precision + np.interp(all_recall, recall[c][::-1], precision[c][::-1]) # xp in interp() must be in increasing order
    # Finally average it and compute AUPRC
    mean_precision = mean_precision/n_classes
    recall["macro"] = all_recall
    precision["macro"] = mean_precision
    #roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(test_set_Y_org.ravel(), test_set_y_pred_prob.ravel())
    average_precision["macro"] = average_precision_score(test_set_Y_org, test_set_y_pred_prob, average="macro") # micro macro, weighted, or samples
    average_precision["micro"] = average_precision_score(test_set_Y_org, test_set_y_pred_prob, average="micro") # micro macro, weighted, or samples
    average_precision["weighted"] = average_precision_score(test_set_Y_org, test_set_y_pred_prob, average="weighted") # micro macro, weighted, or samples
    average_precision["samples"] = average_precision_score(test_set_Y_org, test_set_y_pred_prob, average="samples") # micro macro, weighted, or samples

    if plot_curve:
        fig=plt.figure(num=1,figsize=(figwidth,figheight))
        ax=fig.add_subplot(1,1,1)
        if n_classes>2 or positive_class_for_two_classes is None:
            ax.plot(recall["macro"], precision["macro"], linewidth=1,color=colors[n_classes],label='macro-avg PRC (area={0:0.4f})'.format(average_precision["macro"]))
        
        for c in range(n_classes):
            if positive_class_for_two_classes==None or (n_classes==2 and positive_class_for_two_classes==c):
                ax.plot(recall[c], precision[c],linewidth=1,color=colors[c],label='PRC of {0} (area={1:0.4f})'.format(classes_unique[c], average_precision[c]))
                
        # add some text for labels, title and axes ticks
        ax.set_ylim(0.0,1.0)
        ax.set_xlim(0.0,1.0)
        ax.set_ylabel("Precision",fontsize=12)
        ax.set_xlabel("Recall",fontsize=12)    
        #ax.set_title("",fontsize=15)
        ax.legend(loc="lower left",fontsize=8)
        #plt.subplots_adjust(bottom=0.12) # may this is not working because of the following setting
        fig.savefig(filename,bbox_inches='tight')
        plt.close(fig)

    average_precision_list=[average_precision[c] for c in range(n_classes)]
    average_precision_list.extend([average_precision["macro"],average_precision["micro"],average_precision["weighted"],average_precision["samples"]])
    average_precision=np.array(average_precision_list)
    names=["AUPRC_" + c for c in classes_unique]
    names.extend(["macro","micro","weighted","samples"])
    names=np.array(names)
      
    return average_precision,names


def roc(test_set_y_org,test_set_y_pred_prob,classes_unique,plot_curve=False,filename="./fig_roc.pdf",colors=None,positive_class_for_two_classes=None,figwidth=5,figheight=5):
    """
    Calcuate the area under ROC, and draw the ROC.
    """
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    from sklearn.metrics import roc_auc_score
    from scipy import interp
    import matplotlib as mpl
    mpl.use("pdf")
    import matplotlib.pyplot as plt
  
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes=len(classes_unique)
    test_set_Y_org,test_set_y_org_unique=membership_vector_to_indicator_matrix(test_set_y_org)
    
    for c in range(n_classes):
        fpr[c], tpr[c], _ = roc_curve(test_set_Y_org[:, c], test_set_y_pred_prob[:, c])
        roc_auc[c] = roc_auc_score(test_set_Y_org[:, c], test_set_y_pred_prob[:, c])
        
    # Compute macro-average ROC curve and AUROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[c] for c in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[c], tpr[c])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    #roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Compute micro-average PRC curve and PRC areas
    fpr["micro"], tpr["micro"], _ = roc_curve(test_set_Y_org.ravel(), test_set_y_pred_prob.ravel())
    roc_auc["macro"] = roc_auc_score(test_set_Y_org, test_set_y_pred_prob, average="macro") # micro macro, weighted, or samples
    roc_auc["micro"] = roc_auc_score(test_set_Y_org, test_set_y_pred_prob,average="micro") # micro macro, weighted, or samples
    roc_auc["weighted"] = roc_auc_score(test_set_Y_org, test_set_y_pred_prob, average="weighted") # micro macro, weighted, or samples
    roc_auc["samples"] = roc_auc_score(test_set_Y_org, test_set_y_pred_prob, average="samples") # micro macro, weighted, or samples

    if plot_curve:
        fig=plt.figure(num=1,figsize=(figwidth,figheight))
        ax=fig.add_subplot(1,1,1)
        ax.plot([0, 1], [0, 1], 'k--')
        if n_classes>2 or positive_class_for_two_classes is None:
            ax.plot(fpr["macro"], tpr["macro"], linewidth=1,color=colors[n_classes],label='macro-avg ROC (area={0:0.4f})'.format(roc_auc["macro"]))
        
        for c in range(n_classes):
            if positive_class_for_two_classes is None or (n_classes==2 and positive_class_for_two_classes==c):
                ax.plot(fpr[c], tpr[c],linewidth=1,color=colors[c],label='ROC of {0} (area={1:0.4f})'.format(classes_unique[c], roc_auc[c]))
                
        # add some text for labels, title and axes ticks
        ax.set_ylim(0.0,1.0)
        ax.set_xlim(0.0,1.0)
        ax.set_ylabel("True Positive Rate",fontsize=12)
        ax.set_xlabel("False Positive Rate",fontsize=12)    
        #ax.set_title("",fontsize=15)
        ax.legend(loc="lower right",fontsize=8)
        #plt.subplots_adjust(bottom=0.12) # may this is not working because of the following setting
        fig.savefig(filename,bbox_inches='tight')
        plt.close(fig)

    roc_auc_list=[roc_auc[c] for c in range(n_classes)]
    roc_auc_list.extend([roc_auc["macro"],roc_auc["micro"],roc_auc["weighted"],roc_auc["samples"]])
    roc_auc=np.array(roc_auc_list)
    names=["AUROC_" + c for c in classes_unique]
    names.extend(["macro","micro","weighted","samples"])
    names=np.array(names)
    return roc_auc,names


def prcs(test_set_y_org,test_set_y_pred_prob,methods,linestyles,classes_unique,plot_curve=False,filename="./fig_prc.pdf",colors=None,positive_class_for_two_classes=None,figwidth=5,figheight=5):
    """
    Calcuate the area under precision-recall curve, and draw the precision-recall curve.
    INPUTS:
    test_set_y_pred_prob: list of numpy 1d arrays.
    methods: list of strings, the classification methods used.
    classes_unique: list or numpy 1d array of strings, the unique class labels.
    linestyles: list of strings, including "solid","dashed", "dashdot", "dotted", each style corresponds to a method.
    """
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from scipy import interp
    import matplotlib as mpl
    mpl.use("pdf")
    import matplotlib.pyplot as plt
    
    n_classes=len(classes_unique)
    test_set_Y_org,test_set_y_org_unique=membership_vector_to_indicator_matrix(test_set_y_org)

    num_methods=len(methods)
    average_precisions=[0]*num_methods
    names=[None]*num_methods
    for m in range(num_methods):
        precision = dict()
        recall = dict()
        average_precision = dict()
        for c in range(n_classes):
            precision[c], recall[c], _ = precision_recall_curve(test_set_Y_org[:, c], test_set_y_pred_prob[m][:, c])
            average_precision[c] = average_precision_score(test_set_Y_org[:, c], test_set_y_pred_prob[m][:, c])

        # Compute macro-average ROC curve and AUROC area
        # First aggregate all recalls
        all_recall = np.unique(np.concatenate([recall[c] for c in range(n_classes)]))
        #all_recall = np.sort(np.concatenate([recall[c] for c in range(n_classes)]))
        # Then interpolate all PRC curves at this points
        mean_precision = np.zeros_like(all_recall)
        for c in range(n_classes):
            mean_precision = mean_precision + np.interp(all_recall, recall[c][::-1], precision[c][::-1]) # xp in interp() must be in increasing order
        # Finally average it and compute AUPRC
        mean_precision = mean_precision/n_classes
        recall["macro"] = all_recall
        precision["macro"] = mean_precision
        #roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        # Compute micro-average ROC curve and ROC area
        precision["micro"], recall["micro"], _ = precision_recall_curve(test_set_Y_org.ravel(), test_set_y_pred_prob[m].ravel())
        average_precision["macro"] = average_precision_score(test_set_Y_org, test_set_y_pred_prob[m], average="macro") # micro macro, weighted, or samples
        average_precision["micro"] = average_precision_score(test_set_Y_org, test_set_y_pred_prob[m], average="micro") # micro macro, weighted, or samples
        average_precision["weighted"] = average_precision_score(test_set_Y_org, test_set_y_pred_prob[m], average="weighted") # micro macro, weighted, or samples
        average_precision["samples"] = average_precision_score(test_set_Y_org, test_set_y_pred_prob[m], average="samples") # micro macro, weighted, or samples

        if plot_curve:
            if m==0:
                fig=plt.figure(num=1,figsize=(figwidth,figheight))
                ax=fig.add_subplot(1,1,1)
            if n_classes>2 or positive_class_for_two_classes is None:
                ax.plot(recall["macro"], precision["macro"], linestyle=linestyles[m],linewidth=1,color=colors[n_classes],label='macro-avg PRC (area={0:0.4f}), {1}'.format(average_precision["macro"], methods[m]))

            for c in range(n_classes):
                if positive_class_for_two_classes==None or (n_classes==2 and positive_class_for_two_classes==c):
                    ax.plot(recall[c], precision[c],linestyle=linestyles[m],linewidth=1,color=colors[c],label='PRC of {0} (area={1:0.4f}), {2}'.format(classes_unique[c], average_precision[c], methods[m]))

            # add some text for labels, title and axes ticks
            if m==num_methods-1:
                ax.set_ylim(0.0,1.0)
                ax.set_xlim(0.0,1.0)
                ax.set_ylabel("Precision",fontsize=12)
                ax.set_xlabel("Recall",fontsize=12)    
                #ax.set_title("",fontsize=15)
                ax.legend(loc="lower left",fontsize=8)
                #plt.subplots_adjust(bottom=0.12) # may this is not working because of the following setting
                fig.savefig(filename,bbox_inches='tight')
                plt.close(fig)

        average_precision_list=[average_precision[c] for c in range(n_classes)]
        average_precision_list.extend([average_precision["macro"],average_precision["micro"],average_precision["weighted"],average_precision["samples"]])
        average_precision=np.array(average_precision_list)
        name=[methods[m]+"_AUPRC_" + c for c in classes_unique]
        name.extend(["macro","micro","weighted","samples"])
        name=np.array(name)

        average_precisions[m]=average_precision
        names[m]=name
      
    return average_precisions,names


def rocs(test_set_y_org,test_set_y_pred_prob,methods,linestyles,classes_unique,plot_curve=False,filename="./fig_roc.pdf",colors=None,positive_class_for_two_classes=None,figwidth=5,figheight=5):
    """
    Calcuate the area under ROC, and draw the ROC.
    linestyles: list of strings, including "solid","dashed", "dashdot", "dotted"
    """
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    from sklearn.metrics import roc_auc_score
    from scipy import interp
    import matplotlib as mpl
    mpl.use("pdf")
    import matplotlib.pyplot as plt
  
    n_classes=len(classes_unique)
    test_set_Y_org,test_set_y_org_unique=membership_vector_to_indicator_matrix(test_set_y_org)

    num_methods=len(methods)
    roc_aucs=[0]*num_methods
    names=[None]*num_methods
    for m in range(num_methods):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for c in range(n_classes):
            fpr[c], tpr[c], _ = roc_curve(test_set_Y_org[:, c], test_set_y_pred_prob[m][:, c])
            roc_auc[c] = roc_auc_score(test_set_Y_org[:, c], test_set_y_pred_prob[m][:, c])

        # Compute macro-average ROC curve and AUROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[c] for c in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for c in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[c], tpr[c])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        #roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Compute micro-average PRC curve and PRC areas
        fpr["micro"], tpr["micro"], _ = roc_curve(test_set_Y_org.ravel(), test_set_y_pred_prob[m].ravel())
        roc_auc["macro"] = roc_auc_score(test_set_Y_org, test_set_y_pred_prob[m], average="macro") # micro macro, weighted, or samples
        roc_auc["micro"] = roc_auc_score(test_set_Y_org, test_set_y_pred_prob[m],average="micro") # micro macro, weighted, or samples
        roc_auc["weighted"] = roc_auc_score(test_set_Y_org, test_set_y_pred_prob[m], average="weighted") # micro macro, weighted, or samples
        roc_auc["samples"] = roc_auc_score(test_set_Y_org, test_set_y_pred_prob[m], average="samples") # micro macro, weighted, or samples

        if plot_curve:
            if m==0:
                fig=plt.figure(num=1,figsize=(figwidth,figheight))
                ax=fig.add_subplot(1,1,1)
            ax.plot([0, 1], [0, 1], 'k--')
            if n_classes>2 or positive_class_for_two_classes is None:
                ax.plot(fpr["macro"], tpr["macro"], linestyle=linestyles[m],linewidth=1,color=colors[n_classes],label='{0}: macro-avg ROC (area={1:0.4f})'.format(methods[m], roc_auc["macro"]))

            for c in range(n_classes):
                if positive_class_for_two_classes is None or (n_classes==2 and positive_class_for_two_classes==c):
                    ax.plot(fpr[c], tpr[c],linestyle=linestyles[m],linewidth=1,color=colors[c],label='{0}: ROC of {1} (area={2:0.4f})'.format(methods[m], classes_unique[c], roc_auc[c]))

            # add some text for labels, title and axes ticks
            if m==num_methods-1:
                ax.set_ylim(0.0,1.0)
                ax.set_xlim(0.0,1.0)
                ax.set_ylabel("True Positive Rate",fontsize=12)
                ax.set_xlabel("False Positive Rate",fontsize=12)    
                #ax.set_title("",fontsize=15)
                ax.legend(loc="lower right",fontsize=8)
                #plt.subplots_adjust(bottom=0.12) # may this is not working because of the following setting
                fig.savefig(filename,bbox_inches='tight')
                plt.close(fig)

        roc_auc_list=[roc_auc[c] for c in range(n_classes)]
        roc_auc_list.extend([roc_auc["macro"],roc_auc["micro"],roc_auc["weighted"],roc_auc["samples"]])
        roc_auc=np.array(roc_auc_list)
        name=[methods[m]+"_AUROC_" + c for c in classes_unique]
        name.extend(["macro","micro","weighted","samples"])
        name=np.array(name)

        roc_aucs[m]=roc_auc
        names[m]=name
        
    return roc_aucs,names
    

def con_mat_to_num(confusion_matrix):
    """
    Compute performance given confusion matrix.
    
    INPUTS:
    confusion_matrix: list or numpy array.
    
    OUTPUTS:
    perf: numpy 1d array or vector of length 2*C+3, 
    [acc_0, acc_1, acc_{C-1}, accuracy, balanced accuracy].
    For two-classes, perf=[sensitivity, specificity, PPV, NPV, accuracy, averaged sensitivity (or called balanced accuracy, that is 0.5*(sen+spec) ), averaged PVs (that is 0.5*(PPV+NPV) )].
    For multi-classes, perf=[sen_0, sen_1, ..., sen_{C-1}, precision_0, precision_1, ..., precision_{C-1}, accuracy, averaged sensitivity, averaged percision] that is [class-wise rates, class-wise predictive rates, accuracy, averaged class-wise rate, averaged class-wise predictive rate].
    """
    confusion_matrix=np.array(confusion_matrix) # convert to numpy array
    numcl=confusion_matrix.shape[0] # number of classes
    perf=np.zeros((2*numcl+3,)) # sensitivity_0,sensitivity_1,...,sensitivity_{C-1}, precision_0,precision_1,...,precision_{C-1}, accuracy, balanced sensitivity, balanced precision 
    perf[0:numcl]=confusion_matrix.diagonal()/confusion_matrix.sum(axis=1) # sensitivity and specifity for two classes, (class-wise rates for multi-classes)
    perf[numcl:2*numcl]=confusion_matrix.diagonal()/confusion_matrix.sum(axis=0) # PPV and NPV for two classes, (class-wise predictive rates for multi-classes)
    perf[2*numcl]=confusion_matrix.diagonal().sum()/confusion_matrix.sum(axis=1).sum() # accuracy
    perf[2*numcl+1]=np.mean(perf[0:numcl]) # balanced accuracy for two-classes, average class-wise rate for multi-class 
    perf[2*numcl+2]=np.mean(perf[numcl:2*numcl]) # avarage class-wise predictive rate
    return perf


def save_all_perfs_aurocs_auprcs(path="./",filename="perfs_aurocs_auprcs.txt",classes_unique=None,num_runs=1,perfs=None,aurocs=None,auprcs=None):
    """
    Save the perfs, auROCs, and auPRCs from all runs.
    classes_unique: list of strings, the unique class names.
    num_runs: integer, number of runs.
    perfs: numpy array, each row corresponds to a run.
    aurocs: numpy array, each row corresponds to a run.
    auprcs: numpy array, each row corresponds to a run.
    """
    try:
        os.makedirs(path)
    except OSError:
        pass
    num_classes=len(classes_unique)
    filename=path + "/" +filename
    classes_unique_row=np.array(classes_unique)
    classes_unique_row.shape=(1,len(classes_unique))
    file_handle=file(filename,'w') # create a new file
    file_handle.close()
    file_handle=file(filename,'a') # open to append
    if perfs is not None:
        if num_classes==2:
            header_perf=["Sensitivity","Specificity","PPV", "NPV", "Accuracy", "Balanced_Accuracy", "Averaged_PVs"]
        else:
            header_perf=["Classwise_Rate_" + c for c in classes_unique]
            header_perf.extend(["Classwise_Predictive_Rate_" + c for c in classes_unique])
            header_perf.extend(["Accuracy","Averaged_Classwise_Rate","Averaged_Classwise_Predictive_Rate"])
        if num_runs==1:
            perfs.shape(1,size(perfs))
    else:
        header_perf=[]
        perfs=np.array([])
        perfs.shape=(num_runs,0)
    if aurocs is not None:
        header_auroc=["AUROC_" + c for c in classes_unique]
        header_auroc.extend(["macro","micro","weighted","samples"])
        if num_runs==1:
            aurocs.shape(1,size(aurocs))
    else:
        header_auroc=[]
        aurocs=np.array([])
        aurocs.shape=(num_runs,0)
    if auprcs is not None:
        header_auprc=["AUPRC_" + c for c in classes_unique]
        header_auprc.extend(["macro","micro","weighted","samples"])
        if num_runs==1:
            auprcs.shape(1,size(auprcs))
    else:
        header_auprc=[]
        auprcs=np.array([])
        auprcs.shape=(num_runs,0)
        
    # save header
    header=[]
    header.extend(header_perf)
    header.extend(header_auroc)
    header.extend(header_auprc)
    header=np.array(header)
    header.shape=(1,len(header))
    np.savetxt(file_handle,header,fmt="%s",delimiter='\t')
    # save all results
    results=np.hstack((perfs,aurocs,auprcs))
    np.savetxt(file_handle,results,fmt="%1.4f",delimiter="\t")
    file_handle.close()
    

def save_perform(path="./",filename="mean_performances.txt",create_new_file=True,perf=None,std=None,auroc=None,auroc_std=None,auprc=None,auprc_std=None,conf_mat=None,classes_unique=None,training_time=None,test_time=None,stat_test=None):
    """
    Save performance to a txt file.
    
    INPUTS:
    path: string, the path information, e.g. path="/home/yifengli/prog/my/DECREAS/result".
    
    filename: string, the name of the txt file to save the performance, e.g. filename="performance.txt".

    perf: numpy 1d array, the performance.

    std: numpy 1d array, STD.
    
    conf_mat: numpy 2d array, confusion matrix.

    classes_unique: numpy 1d array, the unique class labels.

    training_time: scalar.
    
    test_time: scalar.
    """
    try:
        os.makedirs(path)
    except OSError:
        pass
    num_classes=len(classes_unique)
    filename=path + "/" +filename
    classes_unique_row=np.array(classes_unique)
    classes_unique_row.shape=(1,len(classes_unique))
    if create_new_file:
        file_handle=file(filename,'w') # create a new file
        file_handle.close()
    file_handle=file(filename,'a') # open to append
    np.savetxt(file_handle,classes_unique_row,fmt="%s",delimiter='\t')
    if perf is not None:
        if num_classes==2:
            header=["Sensitivity","Specificity","PPV", "NPV", "Accuracy", "Balanced_Accuracy", "Averaged_PVs"]
        else:
            header=["Classwise_Rate_" + c for c in classes_unique]
            header.extend(["Classwise_Predictive_Rate_" + c for c in classes_unique])
            header.extend(["Accuracy","Averaged_Classwise_Rate","Averaged_Classwise_Predictive_Rate"])
        header=np.array(header)
        header.shape=(1,len(header))
        np.savetxt(file_handle,header,fmt="%s",delimiter='\t')
        perf=np.array(perf)
        perf.shape=(1,len(perf))
        np.savetxt(file_handle,perf,fmt="%1.4f",delimiter="\t")
    if std is not None:
        std=np.array(std)
        std.shape=(1,len(std))
        np.savetxt(file_handle,std,fmt="%1.4f",delimiter="\t")
    if auroc is not None:
        np.savetxt(file_handle,["AUROC"],fmt="%s",delimiter="\t")
        header=["AUROC_" + c for c in classes_unique]
        header.extend(["macro","micro","weighted","samples"])
        header=np.array(header)
        header.shape=(1,len(header))
        np.savetxt(file_handle,header,fmt="%s",delimiter='\t')
        auroc=np.array(auroc)
        auroc.shape=(1,len(auroc))
        np.savetxt(file_handle,auroc,fmt="%1.4f",delimiter="\t")
    if auroc_std is not None:
        auroc_std=np.array(auroc_std)
        auroc_std.shape=(1,len(auroc_std))
        np.savetxt(file_handle,auroc_std,fmt="%1.4f",delimiter="\t")
    if auprc is not None:
        np.savetxt(file_handle,["AUPRC"],fmt="%s",delimiter="\t")
        header=["AUPRC_" + c for c in classes_unique]
        header.extend(["macro","micro","weighted","samples"])
        header=np.array(header)
        header.shape=(1,len(header))
        np.savetxt(file_handle,header,fmt="%s",delimiter='\t')
        auprc=np.array(auprc)
        auprc.shape=(1,len(auprc))
        np.savetxt(file_handle,auprc,fmt="%1.4f",delimiter="\t")
    if auprc_std is not None:
        auprc_std=np.array(auprc_std)
        auprc_std.shape=(1,len(auprc_std))
        np.savetxt(file_handle,auprc_std,fmt="%1.4f",delimiter="\t")
    if conf_mat is not None:
        np.savetxt(file_handle,["Confusion Matrix"],fmt="%s",delimiter="\t")
        np.savetxt(file_handle,classes_unique_row,fmt="%s",delimiter="\t")
        np.savetxt(file_handle,conf_mat,fmt="%d",delimiter="\t")
    if training_time is not None:
        np.savetxt(file_handle,["Training_Time"],fmt="%s",delimiter="\t")
        np.savetxt(file_handle,np.array([training_time]),fmt="%1.4e",delimiter="\t")
    if test_time is not None:
        np.savetxt(file_handle,["Test_Time"],fmt="%s",delimiter="\t")
        np.savetxt(file_handle,np.array([test_time]),fmt="%1.4e",delimiter="\t")
    if stat_test is not None:
        np.savetxt(file_handle,["Statistical_Test"],fmt="%s",delimiter="\t")
        np.savetxt(file_handle,np.array([stat_test]),fmt="%1.4e",delimiter="\t")
        
    #if training_time is not None and test_time is not None:
    #    np.savetxt(file_handle,["Training_Time"],fmt="%s",delimiter="\t")
    #    np.savetxt(file_handle,np.array([training_time,test_time]),fmt="%1.4e",delimiter="\t")
    #if training_time is not None and test_time is None:
    #    np.savetxt(file_handle,np.array(training_time),fmt="%1.4e",delimiter="\t")
    #if training_time is None and test_time is not None:
    #    np.savetxt(file_handle,np.array(test_time),fmt="%1.4e",delimiter="\t")
    #np.savetxt(file_handle,np.array(test_time),fmt="%s",delimiter="\t")
    file_handle.close()
    

def save_perform_old(path,filename,perf=None,std=None,conf_mat=None,classes_unique=None,training_time=None,test_time=None):
    """
    Save performance to a txt file.
    
    INPUTS:
    path: string, the path information, e.g. path="/home/yifengli/prog/my/DECREAS/result".
    
    filename: string, the name of the txt file to save the performance, e.g. filename="performance.txt".

    perf: numpy 1d array, the performance.

    std: numpy 1d array, STD.
    
    conf_mat: numpy 2d array, confusion matrix.

    classes_unique: numpy 1d array, the unique class labels.

    training_time: scalar.
    
    test_time: scalar.
    """
    try:
        os.makedirs(path)
    except OSError:
        pass
    filename=path + "/" +filename
    np.savetxt(filename,classes_unique,fmt="%s",delimiter='\t')
    file_handle=file(filename,'a')
    if perf is not None:
        np.savetxt(file_handle,perf,fmt="%1.4f",delimiter="\t")
    if std is not None:
        np.savetxt(file_handle,std,fmt="%1.4f",delimiter="\t")
    if conf_mat is not None:
        np.savetxt(file_handle,conf_mat,fmt="%d",delimiter="\t")
    if training_time is not None and test_time is not None:
        np.savetxt(file_handle,np.array([training_time,test_time]),fmt="%1.4e",delimiter="\t")
    if training_time is not None and test_time is None:
        np.savetxt(file_handle,np.array(training_time),fmt="%1.4e",delimiter="\t")
    if training_time is None and test_time is not None:
        np.savetxt(file_handle,np.array(test_time),fmt="%1.4e",delimiter="\t")
    #np.savetxt(file_handle,np.array(test_time),fmt="%s",delimiter="\t")
    file_handle.close()


def change_max_num_epoch_change_learning_rate(max_num_epoch_change_learning_rate,max_num_epoch_change_rate):
    max_num_epoch_change_learning_rate= int(math.ceil(max_num_epoch_change_rate * max_num_epoch_change_learning_rate))
    if max_num_epoch_change_learning_rate<=20:
        max_num_epoch_change_learning_rate=20
    return max_num_epoch_change_learning_rate    


def drange(start, stop, step):
    """
    Generate a sequences of numbers.
    """
    values=[]
    r = start
    while r <= stop:
        values.append(r)
        r += step
    return values 

    
def write_feature_weight(weights,features,lambda1s,filename):
    """
    Write the weights of the input layer of a DFS model to a file. Only applicable to deep feature selection.
    
    INPUTS:
    weights: numpy 2d array or matrix, 
    rows corresponding to values of lambda1s,
    columns corresponding to features.
    
    features: numpy 1d array or vector, names of features.
    
    lambda1s: numpy 1d array or vector, values of lambda1s.
    
    filename: string, file name to be written.
    
    OUTPUTS: 
    None.
    """
    # example:
    #weights=np.asarray([[1.1,2.2,3.4],[5.5,6.6,7.7]])
    #features=np.asarray(['f1','f2','f3'],dtype=object)
    #lambda1s=np.asarray([1.0,2.0])
    #write_feature_weight(weights,features,lambda1s,filename='test.txt')
    
    features=np.insert(features,0,'lambda')
    weights=np.asarray(weights,dtype=object)
    lambda1s=np.asanyarray(lambda1s,dtype=object)
    lambda1s.resize((lambda1s.shape[0],1))
    lambda1s_weights=np.hstack((lambda1s,weights))
    features.resize((1,features.shape[0]))
    features_lambda1s_weights=np.vstack((features,lambda1s_weights))
    np.savetxt(filename,features_lambda1s_weights,fmt='%s',delimiter='\t')


def write_feature_weight2(weights=None, features=None, lambda1s=None, accuracy=None, uniqueness=False, tol=1e-4, filename='selected_features.txt',many_features=False):
    """
    Write the weights of the input layer of a DFS and other information (accuracy, feature subsets) to a file.  Only applicable to deep feature selection.
    
    INPUTS:
    weights: numpy 2d array or matrix, 
    rows corresponding to values of lambda1,
    columns corresponding to features.
    
    features: numpy 1d array or vector, name of features.
    
    lambda1s: numpy 1d array or vector, values of lambda1.
    
    accuracy: numpy 1d array or vector, accuracy corresponding to each lambda1.
    This parameter is optional.
    
    uniqueness: bool, indiates if only writing unique sizes of feature subsets.
    
    tol: threshold, weights below tol*w_max are considered to be zeros.
    
    filename: string, file name to be written.
    
    OUTPUTS: 
    The output file is arranged as [lambda,accuracy,num_selected,feature_subset,weights_of_feature_subset]
    """
    weights=np.asarray(weights,dtype=float)
    lambda1s=np.asarray(lambda1s,dtype=float)
    num_selected=np.zeros(len(lambda1s),dtype=int) # for each lambda, save the number of selected features
    features_selected=np.zeros(len(lambda1s),dtype=object)    
    # get the numbers of selected features
    for i in range(len(lambda1s)):
        w=weights[i]
        w_max=np.max(abs(w))
        w_min=np.min(abs(w))
        if tol*w_max<=w_min: # there is no element that is much larger: either none selected, or select all
            continue
        selected=(abs(w)>tol*w_max)
        #selected=(abs(w)>tol)
        num_selected[i]=selected.sum()
        feat_selected=features[selected]
        w_selected=w[selected]
        ind=np.argsort(abs(w_selected))
        ind=ind[::-1]
        feat_selected=feat_selected[ind]
        features_selected[i]=','.join(feat_selected)
        
    # take the first non-zeros
    if uniqueness:
        if accuracy is not None:
           _,_,take=take_max(num_selected,accuracy)
        else:
            take=take_first(num_selected)
    else:
        take=np.ones(len(num_selected),dtype=bool)
    weights_take=weights[take]
    lambda1s_take=lambda1s[take]
    lambda1s_take.resize((lambda1s_take.shape[0],1))
    lambda1s_take.round(decimals=6)
    features_take=features_selected[take]
    features_take.resize((features_take.shape[0],1))
    num_take=num_selected[take]
    # if no subset is selected
    if num_take.shape[0]==0:
        return None         
    # if the last one is zero, then it means that all features are selected
    if num_take.shape[0]>1 and num_take[-1]==0 and num_take[-2]>0:
        num_take[-1]=len(features)
        features_take[-1]=','.join(features)    
    num_take.resize((num_take.shape[0],1))
    
    if accuracy is not None:
        accuracy=np.asarray(accuracy,dtype=float)
        accuracy_take=accuracy[take]
        accuracy_take.resize((accuracy_take.shape[0],1))
        accuracy_take.round(decimals=4)
        features=np.insert(features,0,['lambda','accuracy','num_selected','feature_subset'])
        features.resize((1,features.shape[0]))
        if not many_features:
            data=np.hstack((lambda1s_take,accuracy_take, num_take,features_take,weights_take))
            data=np.vstack((features,data))
	else:
	    header=np.array(['lambda','accuracy','num_selected'])
            header.resize((1,header.shape[0]))
	    data=np.hstack((lambda1s_take,accuracy_take, num_take))
	    data=np.vstack((header,data))
    else:
	if not many_features:
            features=np.insert(features,0,['lambda','num_selected','feature_subset'])
            features.resize((1,features.shape[0]))
            data=np.hstack((lambda1s_take,num_take,features_take,weights_take))
            data=np.vstack((features,data))
	else:
            header=np.array(['lambda','num_selected'])
            header.resize((1,header.shape[0]))
	    data=np.hstack((lambda1s_take, num_take))
	    data=np.vstack((header,data))

    np.savetxt(filename,data,fmt='%s',delimiter='\t')

   
def take_first(nums):
    """
    Return the first distinct nonzeros.
    Yifeng Li in UBC.
    Aug 30, 2014.
    Example:
    nums=[0,0,0,1,2,2,2,3,4,4,5,5,5,5,6,7,7,8]
    take_first(nums)
    """
    take=np.zeros(len(nums),dtype=bool)
    if len(nums)==1:
        if nums[0]!=0:
            take[0]=True
        return take
    i=0
    while i<len(nums)-1:
        if nums[i]==0:
            i=i+1
            continue
        if i==0 and nums[i]==nums[i+1]:
            take[i]=True
        if i>0 and nums[i-1]==0:
            take[i]=True
        if i==0 and nums[i] != nums[i+1]:
            take[i]=True
            take[i+1]=True
        if nums[i] != nums[i+1]:
            take[i+1]=True
        i=i+1    
    return take


def take_max(num_feat,acc):
    num_feat=np.array(num_feat,dtype=int)
    acc=np.array(acc,dtype=float)
    indices_num=np.arange(len(num_feat))
    us=np.unique(num_feat)
    num_feat_max=[]
    acc_max=[]
    indices_num_max=[]
    for u in us:
        ind=num_feat==u
        num_feat_this=num_feat[ind]
        acc_this=acc[ind]
        indices_num_this=indices_num[ind]
        max_ind=np.argmax(acc_this)
        num_feat_max.extend([u])
        acc_max.extend([acc_this[max_ind]])
        indices_num_max.extend([indices_num_this[max_ind]])
        
    return np.array(num_feat_max,dtype=int),np.array(acc_max,dtype=float),np.array(indices_num_max,dtype=int)


def take_max_acc_for_each_feature_size(num_feat,acc,feat_subset):

    num_feat=np.array(num_feat,dtype=int)
    acc=np.array(acc,dtype=float)
    feat_subset=np.array(feat_subset,dtype=object)
    us=np.unique(num_feat)
    num_feat_max=[]
    acc_max=[]
    feat_subset_max=[]
    for u in us:
        ind=num_feat==u
        num_feat_this=num_feat[ind]
        acc_this=acc[ind]
        feat_subset_this=feat_subset[ind]
        max_ind=np.argmax(acc_this)
        num_feat_max.extend([u])
        acc_max.extend([acc_this[max_ind]])
        feat_subset_max.extend([feat_subset_this[max_ind]])
        
    return np.array(num_feat_max,dtype=int),np.array(acc_max,dtype=float),np.array(feat_subset_max,dtype=object)


def reduce_sample_size(data,classes,times=2):
    """
    Reduce sample by to 1/times.
    
    INPUTS: 
    data: numpy 2d array or matrix, each row is a sample, the original data.
    
    classes: numpy 1d array or vector, class labels, the original labels.
    
    times: int.
    
    OUTPUTS:
    data: the reduced data.
    
    clases: the reduced classes.
    """
    data=data[range(0,data.shape[0],times)]
    classes=classes[range(0,classes.shape[0],times)]
    return data,classes


def take_some_features(data,features,given=None):
    """
    Use a subset of given features for vectoral samples.
    INPUTS: 
    data: numpy 2d array or matrix, each row is a sample, the original data.

    features: numpy 1d array for features.

    given: numpy 1d array or list for features to be used. given=None will use all features.

    OUTPUTS:
    data: numpy 2d array or matrix, the data using given features.

    features: numpy 1d array, used features.
    """
    if given is None:
        return data,features
    common,ind1,ind2=take_common_features(features,given)
    data=data[:,ind1]
    features=features[ind1]
    return data,features


def exclude_some_features(data,features,given=None):
    """
    Exclude some features for vectoral samples.
    INPUTS: 
    data: numpy 2d array or matrix, each row is a sample, the original data.

    features: numpy 1d array for features.

    given: numpy 1d array or list for features to be excluded. given=None will use all features.

    OUTPUTS:
    data: numpy 2d array or matrix, the data excluding given features.

    features: numpy 1d array, remaining features.
    """
    if given is None:
        return data,features
    common,ind1,ind2=take_common_features(features,given)
    data=np.delete(data,ind1,axis=1)
    features=np.delete(features,ind1)
    return data,features


def take_some_features_matrical_samples(data,features,given=None):
    """
    Use a subset of given features for matrical samples.
    """
    num_sample=data.shape[0]
    feat_total=data.shape[1]
    num_signal=len(features)
    feat_each=feat_total//num_signal
    if given is None:
        return data,features
    common,ind1,ind2=take_common_features(features,given)
    data=data.reshape((num_sample,num_signal,feat_each))
    data=data[:,ind1,:]
    features=features[ind1]
    data=data.reshape((num_sample,len(features)*feat_each))
    return data,features


def exclude_some_features_matrical_samples(data,features,given=None):
    """
    Exclude some features for matrical samples.
    """
    num_sample=data.shape[0]
    feat_total=data.shape[1]
    num_signal=len(features)
    feat_each=feat_total//num_signal
    if given is None:
        return data,features
    common,ind1,ind2=take_common_features(features,given)
    data=data.reshape((num_sample,num_signal,feat_each))
    data=np.delete(data,ind1,axis=1)
    features=np.delete(features,ind1)
    data=data.reshape((num_sample,(len(features))*feat_each))
    return data,features


def take_unique_features(data,features,rm_features=None):
    """
    Take unqiue features and make the change in the data accordingly.

    INPUTS:
    data: numpy 2d array or matrix, each row is a sample, the original data.

    features: numpy 1d array for features.
    rm_features: numpy 1d array of strings or list of strings, the features to be removed.

    OUTPUTS:
    data: the data with unique sorted features.

    features: the unique sorted features.
    """
    unik,ind=np.unique(features,return_index=True)
    features=unik
    # remove unwanted features
    if rm_features is not None:
	ind_keep=np.array([True]*len(features))
	for rmf in rm_features:
	    ind_keep[features==rmf]=False
	features=features[ind_keep]
	ind=ind[ind_keep]
    data=data[:,ind]
    return data,features


def take_unique_features_large(filename_data,filename_features,filename_data_save,filename_features_save,rm_features=None,block_size=1000):
    """
    Take unqiue features and make the change in a big data accordingly. Write the resulted data into a txt file.
    """
    # read the features from file
    features_org=np.loadtxt(filename_features,delimiter='\t',dtype=object)
        
    # create a new file to save processed data
    filename_data_save_handle=file(filename_data_save,'w')
    filename_data_save_handle.close()
    # open the new file to save data sequentially
    filename_data_save_handle=file(filename_data_save,'a')
        
    filename_data_handle=open(filename_data,'r')

    count=0
    start=0
    data_block=[]
    end_of_file=False
    print "Start processing ..."
    while not end_of_file:
        line=filename_data_handle.readline()
        if line=='':
            end_of_file=True
        else:
            if start==0:
                data_block=[]
            # remove "\n" at the end
            data_line=line[0:-1]
            # split the string to substrings
            data_line=data_line.split('\t')
            # append the current line to the block  
            data_block.append(data_line)
            # increase total count
            count=count+1
            # get a full block or partial block at the end
        if start==block_size-1 or (end_of_file and start!=0):
            print "processing the %d-th line ..." %count
            
            ### process the block ###
            data_block=np.array(data_block,dtype=str)
            data_block,features=take_unique_features(data_block,features_org,rm_features)
            # append to file
            np.savetxt(filename_data_save_handle,data_block,fmt='%s',delimiter='\t')
            ### finished processing the block ###
                
            # reset the counts of lines in the block (0-based)
            start=0
        else:
            start=start+1
    filename_data_handle.close()            
    filename_data_save_handle.close()
    print "Done! %d lines are processed." %count
    print "The features are:"
    print features

    # save feature list
    np.savetxt(filename_features_save,features,fmt='%s',delimiter='\t')


def take_common_features(feat1,feat2):
    """
    Return common features and their indices.
    
    INPUTS:
    feat1: numpy 1d array, feature set 1
    feat2: numpy 1d array, feature set 2

    OUTPUTS:
    common: numpy 1d array, the common features.
    ind1: numpy 1d array, the indices of the common features in feature set 1.
    ind2: numpy 1d array, the indices of the common features in feature set 2.
    """
    common=np.intersect1d(feat1,feat2) # sorted
    ind1=find_indices(common,feat1)
    ind2=find_indices(common,feat2)
    return common,ind1,ind2


def find_indices(subset,fullset):
    """
    Find the indices of a subset in the fullset. If an element of subset is not in fullset, its index in fullset will be -1.

    Example: 
    # subset=np.array(["f1","f2","f3"])
    # fullset=np.array(["f7","f5","f2","f4","f6","f3","f1"])
    # indices=find_indices(subset,fullset)
    """
    nsub=len(subset)
    indices=-np.ones(subset.shape,dtype=int)
    indices_full=np.arange(0,len(fullset),1,dtype=int)
    for s in range(0,nsub):
        indices[s]=indices_full[fullset==subset[s]] # numerical indices
    return indices


def plot_bar(filename, data, std=None, xlab='x', ylab='y', ylim=[0,1],yticks=np.arange(0,1.1,0.1), title='Bar-Plot', methods=None, datasets=None, figwidth=8, figheight=6, colors=None, legend_loc="lower center", xytick_fontsize=12, xylabel_fontsize=15, title_fontsize=15, legend_fontsize=12):
    """
    Plot grouped bars given a vector.
    data: 1d-array, each element represents the result of a method.
    """
    import matplotlib as mpl
    mpl.use("pdf")
    import matplotlib.pyplot as plt

    data=np.array(data)
    num_methods=len(data)
    
    # colors
    if colors is None:
        colors=['b','r','g','c','m','y','k','w'] # maximally 8 colors allowed so far

    ind = np.arange(num_methods)  # the x locations of the bars
    width = 0.8                   # the width of the bars
    fig=plt.figure(num=1,figsize=(figwidth,figheight))
    ax=fig.add_subplot(1,1,1)
    if std is None:
        ax.bar(ind,data,width,color=colors[0:num_methods],ecolor='k')
    else:
        ax.bar(ind,data,width,color=colors[0:num_methods],yerr=std,ecolor='k')

    # add some text for labels, title and axes ticks
    ax.set_ylabel(ylab,fontsize=xylabel_fontsize)
    ax.set_xlabel(xlab,fontsize=xylabel_fontsize)
    ax.set_title(title,fontsize=title_fontsize)
    ax.set_xticks(ind+0.5*width)
    ax.set_xticklabels( methods )
    #if ylim is None:
    #    yticks=np.arange(0,1.1,0.1)
    #	ylim=[0,1]
    if yticks is not None:
        ax.set_yticks(yticks)
    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])
    plt.setp(ax.get_xticklabels(), fontsize=xytick_fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=xytick_fontsize)
    # shrink axis box    
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #ax.legend( method_bar, methods, loc='lower left', bbox_to_anchor=(1.0, 0.3), fontsize=legend_fontsize )
    #ax.legend(methods, loc=legend_loc, fontsize=legend_fontsize )
    #plt.show()
    fig.savefig(filename,bbox_inches='tight')
    plt.close(fig)


def plot_bar_group(filename, data, std=None, xlab='x', ylab='y', title='Bar-Plot', methods=None, hatchs=None, datasets=None, figwidth=8, figheight=6, colors=None, legend_loc="lower left", xytick_fontsize=12, xylabel_fontsize=15, title_fontsize=15, legend_fontsize=12,ymin=0,ymax=1,rotation=45):
    """
    Plot grouped bars given a matrix. Group by datasets.
    data: 2d-array, #methods X #datasets, each row represents the result of a method on multiple data sets.
    """
    import matplotlib as mpl
    mpl.use("pdf")
    import matplotlib.pyplot as plt
    data=np.array(data)
    num_methods,num_datasets=data.shape
    
    if hatchs is None:
        hatchs=[None]*len(methods)

    # colors
    if colors is None:
        colors=['b','r','g','c','m','y','k','w'] # maximally 8 colors allowed so far

    ind = np.arange(num_datasets)  # the x locations for the groups
    width = 0.8*(1.0/num_methods)       # the width of the bars
    method_bar=[]
    fig=plt.figure(num=1,figsize=(figwidth,figheight))
    ax=fig.add_subplot(1,1,1)
    #fig, ax = plt.subplots()
    for i in range(num_methods):
        if std is None:
            method_bar.append( ax.bar(ind+i*width, data[i,:], width, color=colors[i], ecolor='k', edgecolor='black', linewidth=0.5, hatch=hatchs[i]))
        else:
            std=np.array(std)
            method_bar.append( ax.bar(ind+i*width, data[i,:], width, color=colors[i], yerr=std[i,:], ecolor='k', edgecolor='black', linewidth=0.5, hatch=hatchs[i]))

    # add some text for labels, title and axes ticks
    ax.set_ylabel(ylab,fontsize=xylabel_fontsize)
    ax.set_xlabel(xlab,fontsize=xylabel_fontsize)
    ax.set_title(title,fontsize=title_fontsize)
    ax.set_xticks(ind+0.5*num_methods*width)
    ax.set_xticklabels(datasets, rotation=rotation)
    ax.set_yticks(np.arange(0,1.1,0.1))
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(-0.5,len(datasets)+1)
    plt.setp(ax.get_xticklabels(), fontsize=xytick_fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=xytick_fontsize)
    # shrink axis box    
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #ax.legend( method_bar, methods, loc='lower left', bbox_to_anchor=(1.0, 0.3), fontsize=legend_fontsize )
    ax.legend( method_bar, methods, loc=legend_loc, fontsize=legend_fontsize )
    #plt.show()
    fig.savefig(filename,bbox_inches='tight')
    plt.close(fig)


def plot_dot_group(filename, data, std=None, xlab='x', ylab='y', title='Dot-Plot', methods=None, markers=None, datasets=None, figwidth=8, figheight=6, colors=None, legend_loc="lower left", xytick_fontsize=12, xylabel_fontsize=15, title_fontsize=15, legend_fontsize=12,ymin=0,ymax=1,rotation=45):
    """
    Plot grouped dots given a matrix. Group by datasets.
    data: 2d-array, #methods X #datasets, each row represents the result of a method on multiple data sets.
    """
    import matplotlib as mpl
    mpl.use("pdf")
    import matplotlib.pyplot as plt
    data=np.array(data)
    num_methods,num_datasets=data.shape
    
    # colors
    if colors is None:
        colors=['b','r','g','c','m','y','k','w'] # maximally 8 colors allowed so far, make it robust later

    if markers is None:
        markers=["s","*","^","+","x","p","d","o","v"] # make it robust later

    ind = np.arange(num_datasets)  # the x locations for the groups
    fig=plt.figure(num=1,figsize=(figwidth,figheight))
    ax=fig.add_subplot(1,1,1)
    for i in range(num_methods):
        ax.plot(ind,data[i,:],color=colors[i],linestyle="", marker=markers[i],markerfacecolor=colors[i],markersize=12)

    # add some text for labels, title and axes ticks
    ax.set_ylabel(ylab,fontsize=xylabel_fontsize)
    ax.set_xlabel(xlab,fontsize=xylabel_fontsize)
    ax.set_title(title,fontsize=title_fontsize)
    ax.set_xticks(ind)
    ax.set_xticklabels(datasets, rotation=rotation)
    ax.set_yticks(np.arange(0,1.1,0.05))
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(ind[0]-0.5,ind[-1]+0.5)
    plt.setp(ax.get_xticklabels(), fontsize=xytick_fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=xytick_fontsize)
    # shrink axis box    
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #ax.legend( method_bar, methods, loc='lower left', bbox_to_anchor=(1.0, 0.3), fontsize=legend_fontsize )
    ax.legend( methods, loc=legend_loc, fontsize=legend_fontsize )
    #plt.show()
    fig.savefig(filename,bbox_inches='tight')
    plt.close(fig)


def plot_3dbar_group(filename, data, std=None, xlab='x', ylab='y', zlab='z', title='3D-Bar-Plot', methods=None, datasets=None, figwidth=4, figheight=3, colors=None, legend_loc="lower left", width=0.5, xytick_fontsize=8, xylabel_fontsize=8, title_fontsize=8, legend_fontsize=8):
    """
    Plot grouped 3d-bars given a matrix.
    data: 2d-array, each row represents the result of a method on multiple data sets.
    """
    import matplotlib as mpl
    mpl.use("pdf")
    import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D
    import mpl_toolkits.mplot3d
    #print mpl.__version__
    #print os.path.abspath(mpl.__file__)

    data=np.array(data)
    num_methods,num_datasets=data.shape

       # colors
    if colors is None:
        colors=['b','r','g','c','m','y','k','w'] # maximally 8 colors allowed so far
    
    fig = plt.figure(figsize=(figwidth,figheight))
    ax = fig.add_subplot(111, projection='3d') # new version
    #ax = Axes3D(fig) # prior version 1.0.0

    for d in range(num_datasets):
        x=np.arange(num_methods)
        y=data[:,d]
        z=d
        color=colors[d]

        ax.bar(left=x, height=y, zs=z, width=width, zdir='y', color=color, alpha=0.8)
        
        #if std is None:
            #ax.bar(left=x, height=y, zs=z, width=0.5, zdir='y', color=color, alpha=0.8)
        #else:
            #std=np.array(std)
            #err=std[:,d]
            #ax.bar(left=x, height=y, zs=z, zdir='y', color=color, alpha=0.8, zerr=err, ecolor='k')
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('\n'+ylab, linespacing=3, fontsize=xylabel_fontsize) # labelpad does not work
    ax.set_xlabel('\n'+xlab, linespacing=2, fontsize=xylabel_fontsize)
    ax.set_zlabel(zlab,fontsize=xylabel_fontsize)
    ax.set_title(title,fontsize=title_fontsize)
    ax.set_xticks(np.arange(num_methods))
    ax.set_xticklabels( methods )
    ax.set_zticks(np.arange(0,1.1,0.1))
    ax.set_ylim(0,num_datasets)   
    ax.set_yticks(np.arange(num_datasets)+1)
    ax.set_yticklabels( datasets )
    ax.set_zlim(0,1)
    plt.setp(ax.get_xticklabels(), fontsize=xytick_fontsize, rotation=-15)
    plt.setp(ax.get_yticklabels(), fontsize=xytick_fontsize)
    plt.setp(ax.get_zticklabels(), fontsize=xytick_fontsize)
    #plt.show()
    fig.savefig(filename)
    plt.close(fig)


def plot_bar_group_subplots(filename, datas, stds=None, xlabs='x', ylabs='y', titles='Bar-Plot', methods=None, datasets=None, figwidth=8, figheight=6, colors=None, legend_loc="lower left", xytick_fontsize=8, xylabel_fontsize=8, title_fontsize=8, legend_fontsize=8, num_col=2, ymin=None, ymax=None):
    """
    Plot subplots of the (classification) performance.
    datas: a list of numpy data matrices, each of which for each subplot.
    stds: a list of numpy error matrices, each of which for each subplot, can be None.
    xlabs: a list of strings, the lable of the x-axis, can be just a string.
    ylabs: a list of strings, the lable of the y-axis, can be just a string.
    titles: a list of strings, the titles of the subplots, can be just a string.
    methods: a list of lists of strings, the legend of each group, can be just a list of strings.
    datasets: a list of lists of the group names, can be just a list of trings.
    figwidth: scalar, width in inches of the figure.
    figheight: scalar, height in inches of the figures.
    colors: colors.
    legend_locs: srting, location of the legend.
    xytick_fontsize: scalar, the font size of the x and y ticks.
    xylable_fontsize: scalar, the font size of the x and y labels.
    legend_fontsize: scalar, the font size of the legend.
    num_col: integer, number of subplots in each column.
    """
    import matplotlib as mpl
    mpl.use("pdf")
    import matplotlib.pyplot as plt

    num_plots=len(datas)
    num_row=int(math.ceil(num_plots/float(num_col)))

    if not isinstance(xlabs,list):
        xlabs=[xlabs]*num_plots
    if not isinstance(ylabs,list):
        ylabs=[ylabs]*num_plots
    if not isinstance(titles,list):
        xlabs=[titles]*num_plots
    if not isinstance(methods[0],list):
        methods=methods*num_plots
    if not isinstance(datasets[0],list):
        datasets=datasets*num_plots
    # colors
    if colors is None:
        colors=['b','r','g','c','m','y','k','w'] # maximally 8 colors allowed so far
    if not isinstance(colors[0],list):
        colors=colors*num_plots

    print colors

    fig,ax=plt.subplots(num_row,num_col,sharex='col')
    fig.set_size_inches(figwidth,figheight)

    for p in range(num_plots):
        # obtain data for current subplot
        data=datas[p]
        data=np.array(data)
        if stds is None:
            std=None
        else:
            std=stds[p]
            std=np.array(std)
        xlab=xlabs[p]
        ylab=ylabs[p]
        title=titles[p]
        print data
        num_methods,num_datasets=data.shape
        ind = np.arange(num_datasets)  # the x locations for the groups
        width = 0.8*(1.0/num_methods)       # the width of the bars
        method_bar=[]
        ax_row=p/num_col
        ax_col=p%num_col
        for i in range(num_methods):
            if std is None:
                method_bar.append( ax[ax_row,ax_col].bar(ind+i*width, data[i,:], width, color=colors[p][i], ecolor='k'))
            else:
                method_bar.append( ax[ax_row,ax_col].bar(ind+i*width, data[i,:], width, color=colors[p][i], yerr=std[i,:], ecolor='k'))

        # add some text for labels, title and axes ticks
        ax[ax_row,ax_col].set_ylabel(ylab,fontsize=xylabel_fontsize)
        ax[ax_row,ax_col].set_xlabel(xlab,fontsize=xylabel_fontsize)    
        ax[ax_row,ax_col].set_title(title,fontsize=title_fontsize)
        ax[ax_row,ax_col].set_xticks(ind+0.5*num_methods*width)
        ax[ax_row,ax_col].set_xticklabels( datasets, rotation=45 )
        ax[ax_row,ax_col].set_yticks(np.arange(0,1.1,0.1))
        ax[ax_row,ax_col].set_ylim(ymin[p],ymax[p])
        plt.setp(ax[ax_row,ax_col].get_xticklabels(), fontsize=xytick_fontsize)
        plt.setp(ax[ax_row,ax_col].get_yticklabels(), fontsize=xytick_fontsize)
        # shrink axis box    
        #box = ax[ax_row,ax_col].get_position()
        #ax[ax_row,ax_col].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #ax[ax_row,ax_col].legend( method_bar, methods, loc='lower left', bbox_to_anchor=(1.0, 0.3), fontsize=legend_fontsize )
        print methods[p]
        ax[ax_row,ax_col].legend( method_bar, methods[p], loc='lower center', fontsize=legend_fontsize )
    #plt.show()
    fig.savefig(filename)
    plt.close(fig)
    
    
def plot_box_multi(filename, data, classes, classes_unique=None,  xlab='x', ylab='y', title='Box-Plot', figwidth=8, figheight=6, ymin=0, ymax=10):
    """
    Plot multiple boxes in a plot according to class information.
    data: 1d-array.
    classes: class information to plot the boxes.
    classes_unique: the unique class labels.
    """
    import matplotlib as mpl
    mpl.use("pdf")
    import matplotlib.pyplot as plt
    data=np.array(data)

    if classes_unique is None:
        class_unique=np.unique(classes)
        
    data_plot=[]
    for cl in classes_unique:
        data_cl=data[classes==cl]
        data_plot.append(data_cl)

    fig=plt.figure(num=1,figsize=(figwidth,figheight))
    ax=fig.add_subplot(1,1,1)
    ax.boxplot(data_plot)
    
    # add some text for labels, title and axes ticks
    ax.set_ylim(ymin,ymax)
    ax.set_ylabel(ylab,fontsize=12)
    ax.set_xlabel(xlab,fontsize=12)    
    ax.set_title(title,fontsize=15)
    ind = np.arange(len(classes_unique))
    #ax.set_xticks(ind)
    ax.set_xticklabels( classes_unique )
    plt.setp(ax.get_xticklabels(), fontsize=12, rotation=90)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    # shrink axis box    
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #ax.legend( method_bar, methods, loc='lower left', bbox_to_anchor=(1.0, 0.3), fontsize=12 )
    #plt.show()
    plt.subplots_adjust(bottom=0.12) # may this is not working because of the following setting
    fig.savefig(filename,bbox_inches='tight')
    plt.close(fig)


def feat_acc_fit(feat_nums,accs,feat_subsets=None,tangent=1):
    """
    Fit the (number_features_selected,accuracy) pairs by tangent, and return fitted parameters, and number of features and corresponding accuracy given tangent.  
    """
    # fit the hyperbolic tangent sigmoid curve
    from scipy.optimize import curve_fit

    #def hyperbolic_tangent_sigmoid(x,k):
    #    return 2/(1+np.exp(-2*k*x))-1
    #popt,pcov=curve_fit(hyperbolic_tangent_sigmoid,feat_nums,accs)
    #k=popt[0]
    #print popt
    
    ## get number of features and corresponding accuracy given a value of tangent
    #if k>0 and k<tangent:
    #    sys.exit()
    ## denote u=exp(-2kx)
    #u_1=2*k-tangent + 2*np.sqrt(k*(k-tangent))
    #u_2=2*k-tangent - 2*np.sqrt(k*(k-tangent))
    #u=None
    #if u_1>0 and u_1<1:
    #    u=u_1
    #    print "u_1 fufil the condition :)"
    #if u_2>0 and u_2<1:
    #    u=u_2
    #    print "u_2 fufil the condition :)"
    #x_tangent=-np.log(u)/(2*k)
    #acc_tangent= hyperbolic_tangent_sigmoid(x_tangent,k)
    #x_for_plot=np.linspace(np.min(feat_nums),np.max(feat_nums),1000)
    #y_for_plot=hyperbolic_tangent_sigmoid(x_for_plot,k)


    def arctan_func(x,k,s):
        return 2*s*np.arctan(k*x)/math.pi
    popt,pcov=curve_fit(arctan_func,feat_nums,accs)
    k=popt[0]
    s=popt[1]
    print popt
    # get number of features and corresponding accuracy given a value of tangent
    if 2*k*s-tangent*math.pi<=0:
        print "error, exit!"
        sys.exit()
    x_tangent=(math.sqrt((2*k*s-tangent*math.pi)/(tangent*math.pi)))/k
    acc_tangent= arctan_func(x_tangent,k,s)
    x_for_plot=np.linspace(np.min(feat_nums),np.max(feat_nums),1000)
    y_for_plot=arctan_func(x_for_plot,k,s)
    return popt,pcov,x_tangent,acc_tangent,x_for_plot,y_for_plot


def plot_3D_surface(X,Y,Z,dir_save="./",prefix="Parameter1_Parameter_2_Accuracy",figwidth=6,figheight=6,xlab="X",ylab="Y",zlab="Z",xyzlabel_fontsize=8,xyztick_fontsize=8,fmt="pdf",dpi=600):
    # X,Y,Z,: 2D numpy array. Z[i,j]=f(X[i,j],Y[i,j]) 
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cbook
    from matplotlib import cm
    from matplotlib.colors import LightSource
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(figwidth,figheight))
    ax = fig.add_subplot(111, projection='3d') # new version
    #fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ls = LightSource(270, 45)
    #rgb = ls.shade(Z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    #surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=False, shade=False)
    surf=ax.plot_surface(X, Y, Z, cmap='hot', cstride=1, rstride=1,linewidth=0.5)
    ax.set_xlabel(xlab,linespacing=3, fontsize=xyzlabel_fontsize)
    ax.set_ylabel(ylab,linespacing=3, fontsize=xyzlabel_fontsize)
    ax.set_zlabel(zlab,linespacing=3, fontsize=xyzlabel_fontsize)
    plt.setp(ax.get_xticklabels(), fontsize=xyztick_fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=xyztick_fontsize)
    plt.setp(ax.get_zticklabels(), fontsize=xyztick_fontsize)
    
    X_unique=np.unique(X)
    ax.set_xticks(X_unique)
    ax.set_xticklabels( np.array(X_unique,dtype=str) )
    ax.set_xlim(np.min(X_unique),np.max(X_unique))
    Y_unique=np.unique(Y)
    ax.set_yticks(Y_unique)
    ax.set_yticklabels( np.array(Y_unique,dtype=str) )
    ax.set_ylim(np.min(Y_unique),np.max(Y_unique))
    ax.set_zticks(np.arange(0,1.1,0.1))	
    ax.set_zlim(0,1)
    #plt.show()
    filename=dir_save+prefix+"_3d_surface."+fmt
    plt.tight_layout()
    fig.savefig(filename,format=fmt,dpi=dpi)
    plt.close(fig)

 
def Bernoulli_sampling(P=0.5,size=None,rng=np.random.RandomState(100)):
    """
    Bernoulli sampling for RMBs and DBMs.
    P: scalar or numpy array, holding the parameter p of Bernoulli distributions.
    size: list or dict, the size of sampled array when P is a scalar, when P is an array, it has the same size of P.
    rng: random number generator.
    """
    if (not np.isscalar(P)):
    	size=P.shape
    if (np.isscalar(P) and size is None):
    	size=1
    S=rng.random_sample(size=size)	
    return np.array(S<P,dtype=int)


def Gaussian_sampling(mu=1,beta=1,size=None,rng=np.random.RandomState(100)):
    """
    Gaussian sampling for RMBs and DBMs.
    mu: scalar or numpy array, holding the mean of Gaussian distributions.
    beta: scalar or numpy array, holding the precision of Gaussian distribution.
    size: list or dict, the size of sampled array when P is a scalar, when P is an array, it has the same size of P.
    rng: random number generator.
    """
    if (not np.isscalar(mu)):
    	size=mu.shape
    	if len(size.shape)==1: # vector
    		size=(len(mu),1) # make it 2d
    		mu.shape=size
    	X=mu
    	for i in range(size[0]):
    		for j in range(size[1]):
    			X[i,j]=rng.normal(mu[i,j],1/math.sqrt(beta[i]),size=1)
    if (np.isscalar(mu) and size is None):
    	size=1
    	X=rng.normal(mu,beta,size=size)
    return X


def sigmoid(X):
	"""
	Compute value of sigmoid function.
	INPUTS:
	X: scalar or numpy array.
	OUTPUTS:
	Numpy array of sigmoid values.
	"""
	if isinstance(X,(list,tuple)):
		X=np.array(X)
	return 1/(1+np.exp(-X))


def data_distribution(X,dir_save="./", prefix="X", figwidth=5, figheight=4, color="red", ymax=None, ylab="Number of Features", qs=range(5,105,5),width=0.02):
    """
    X: numpy array of binary values, each row is a feature.
    """
    import matplotlib as mpl
    mpl.use("pdf")
    import matplotlib.pyplot as plt
    
    rowmeanX=np.mean(X,axis=1)
   
    ##percentile
    #q=range(0,101,5)
    #pctl=numpy.percentile(rowmeanX,q)
    #print pctl
    
    # plot histogram
    fig=plt.figure(num=1,figsize=(figwidth,figheight))
    ax=fig.add_subplot(1,1,1)
        
    ax.hist(rowmeanX, bins=100, range=None, normed=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical',color=color)
    ax.set_ylabel(ylab,fontsize=10)
    ax.set_xlabel("Frequency in Samples",fontsize=10)
    if ymax is not None:
        ax.set_ylim(0,ymax)
    
    filename=dir_save + "fig_hist_" + prefix + ".pdf"
    fig.savefig(filename,bbox_inches='tight')
    plt.close(fig)
    
    # plot bar plot
    #print "Thresholding..."
    #qs=range(5,105,5)
    qs=np.array(qs)
    qs=qs/100
    counts=[]
    for q in qs:
        print q
        cnt=np.sum(rowmeanX>q)
        counts.extend([cnt])

    fig=plt.figure(num=1,figsize=(figwidth,figheight))
    ax=fig.add_subplot(1,1,1)
        
    ax.bar(qs-width/2.0, counts, width=width, bottom=None, color=color)
    ax.set_ylabel(ylab + " Above Threshold",fontsize=10)
    ax.set_xlabel("Threshold of Frequency in Samples",fontsize=10)
    #ax.set_ylim(0,20000)
    
    filename=dir_save + "fig_hist_" + prefix + "_threshold.pdf"
    fig.savefig(filename,bbox_inches='tight')
    plt.close(fig)


