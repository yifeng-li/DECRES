def test_model_piecewise(classifier, filename, batch_size):
    """
    same as test_model but reads input file line by line
    Predict class labels of given data using the model learned.
    
    INPUTS:
    classifier_trained: object of MLP, the model learned by function "train_model". 
    
    filename: file location of txt-file numpy 2d array, each row is a sample whose label to be predicted.
    
    batch_size: int scalar, batch size, efficient for a very large number of test samples.
    
    OUTPUTS:
    test_set_y_predicted: numpy int vector, the class labels predicted.
    """
    k=1
    
    f=1
    for line in open(filename):
        if k==1:
            temp=numpy.fromstring(line, dtype=float, sep='\t')
            k+=1
        else:        
            if k%batch_size==0:
                k=1
                temp2=numpy.fromstring(line, dtype=float, sep='\t')
                temp=numpy.vstack([temp,temp2])
                tempout=test_model(classifier, temp, batch_size)
                if f==1:
                    test_set_y_pred=tempout
                    
                    f=0
                else:
                    test_set_y_pred=numpy.append(test_set_y_pred,tempout)
            else:
                temp2=numpy.fromstring(line, dtype=float, sep='\t')
                temp=numpy.vstack([temp,temp2])
                k+=1

    tempout=test_model(classifier, temp, batch_size)
    test_set_y_pred=numpy.append(test_set_y_pred,tempout)
    return test_set_y_pred
