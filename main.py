#Libraries, imports
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import multivariate_normal as norm
import time
#Function for unpacking the data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
start_time = time.time()
#Unpacking the data files, normalizing them and saving them as variables
trainfile = "data/data_batch_1" #Path to training data file
traindict=unpickle(trainfile)
trainlabelsl=traindict[b'labels']
trainlabels=np.array(trainlabelsl)
traindata=traindict[b'data']
traindata = traindata/255

testfile = "data/test_batch" #Path to testing data file
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
#End of functions from exercises

n_components = 9 #Number of components for PCA and LDA, 
#The number has to be smaller than the number of classes if using LDA

#Using the functions from exercises
#Dimensionality reduction - PCA
pca = PCA(n_components = n_components)
trn_pca_set = pca.fit_transform(traindata)
tst_pca_set = pca.transform(testdata)

#Dimensionality reduction - LDA
lda = LDA(n_components = n_components)
trn_lda_set = lda.fit_transform(traindata, trainlabels)
tst_lda_set = lda.transform(testdata)

#Estimating the distribution parameters for PCA and LDA
pca_means, pca_covs = est_params(trn_pca_set, trainlabels)
lda_means, lda_covs = est_params(trn_lda_set, trainlabels)

#Predictions for both PCA and LDA
pca_pred = predict(tst_pca_set, pca_means, pca_covs)
lda_pred = predict(tst_lda_set, lda_means, lda_covs)

#Computing the accuracy of predictions
pca_acc = np.sum(pca_pred == testlabels)/len(testlabels) * 100
lda_acc = np.sum(lda_pred == testlabels)/len(testlabels) * 100
end_time = time.time() - start_time     #Compute the training/fiting time in seconds
print("Running time: {:2.0f}m{:2.0f}s".format(end_time//60, end_time%60))
print("PCA Accuracy: {:.2f}".format(pca_acc))
print("LDA Accuracy: {:.2f}".format(lda_acc))



                     