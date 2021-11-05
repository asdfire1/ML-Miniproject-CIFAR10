import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal as norm
import time
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

start_time = time.time()
trainfile = "data/data_batch_1"
traindict=unpickle(trainfile)
trainlabelsl=traindict[b'labels']
trainlabels1=np.array(trainlabelsl)
traindata1=traindict[b'data']
del traindict
del trainlabelsl
traindata1 = traindata1/255

trainfile2 = "data/data_batch_2"
traindict2=unpickle(trainfile2)
trainlabelsl2=traindict2[b'labels']
trainlabels2=np.array(trainlabelsl2)
traindata2=traindict2[b'data']
del traindict2
del trainlabelsl2
traindata2 = traindata2/255

trainfile3 = "data/data_batch_3"
traindict3=unpickle(trainfile3)
trainlabelsl3=traindict3[b'labels']
trainlabels3=np.array(trainlabelsl3)
traindata3=traindict3[b'data']
del traindict3
del trainlabelsl3
traindata3 = traindata3/255

trainfile4 = "data/data_batch_4"
traindict4=unpickle(trainfile4)
trainlabelsl4=traindict4[b'labels']
trainlabels4=np.array(trainlabelsl4)
traindata4=traindict4[b'data']
del traindict4
del trainlabelsl4
traindata4 = traindata4/255

trainfile5 = "data/data_batch_5"
traindict5=unpickle(trainfile5)
trainlabelsl5=traindict5[b'labels']
trainlabels5=np.array(trainlabelsl5)
traindata5=traindict5[b'data']
del traindict5
del trainlabelsl5
traindata5 = traindata5/255

traindata=np.concatenate((traindata1, traindata2, traindata3, traindata4, traindata5))
trainlabels=np.concatenate((trainlabels1,trainlabels2, trainlabels3, trainlabels4, trainlabels5))



testfile = "data/test_batch"
testdict=unpickle(testfile)
testlabelsl=testdict[b'labels']
testlabels=np.array(testlabelsl)
testdata=testdict[b'data']
testdata = testdata/255

#Functions from the exercises
def est_params(trn_set, trn_targets):
    '''
    Function for estimating the parameters for multiple gaussian distributions.

    Parameters
    ----------
    trn_set : numpy.ndarray
        Training set.
    trn_targets : numpy.ndarray
        Training targets / class labels.

    Returns
    -------
    means : numpy.ndarray
        Mean vectors for each class.
    covs : numpy.ndarray
        Covariance matrices for each class.
    '''
    #Get the data dimension
    _, d = trn_set.shape 
    #Zero arrays for storing means and covs.
    means = np.zeros((10,d))
    covs = np.zeros((10, d, d))
    
    #For each class compute mean and cov.
    for i in range(10):
        indx = trn_targets == i
        means[i] = np.mean(trn_set[indx], axis = 0)
        covs[i] = np.cov(trn_set[indx].T)
    return means, covs

def predict(tst_set, means, covs):
    '''
    Function for making the class prediction based on maximum likelihood.

    Parameters
    ----------
    tst_set : numpy.ndarray
        Test set.
    means : numpy.ndarray
        Mean vectors for each class.
    covs : numpy.ndarray
        Covariance matrices for each class.
        
    Returns
    -------
    preds : numpy.ndarray
        Class predictions.
    '''
    probs = []
    for i in range(len(covs)):
        probs.append(norm.pdf(tst_set, means[i], covs[i]))
    probs = np.c_[tuple(probs)]
    preds = np.argmax(probs, axis = 1)
    return preds
#End of functions from the exercises

n_components = 190

#Using the functions
#print(traindata)
#PCA
pca = PCA(n_components = n_components)
trn_pca_set = pca.fit_transform(traindata)
tst_pca_set = pca.transform(testdata)


#Compute the distribution parameters for PCA
pca_means, pca_covs = est_params(trn_pca_set, trainlabels)


#Classify the data
pca_pred = predict(tst_pca_set, pca_means, pca_covs)


#Calculating accuraccy
pca_acc = np.sum(pca_pred == testlabels)/len(testlabels) * 100
end_time = time.time() - start_time     #Compute the training/fiting time in seconds
print("Running time: {:2.0f}m{:2.0f}s".format(end_time//60, end_time%60))
print("PCA Accuracy: {:.2f}".format(pca_acc))


                     