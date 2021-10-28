import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import multivariate_normal as norm

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


trainfile = "data/data_batch_1"
traindict=unpickle(trainfile)
trainlabelsl=traindict[b'labels']
trainlabels1=np.array(trainlabelsl)
traindata1=traindict[b'data']
traindata1 = traindata1/255

trainfile2 = "data/data_batch_2"
traindict2=unpickle(trainfile2)
trainlabelsl2=traindict[b'labels']
trainlabels2=np.array(trainlabelsl2)
traindata2=traindict2[b'data']
traindata2 = traindata2/255
'''
trainfile3 = "data/data_batch_3"
traindict3=unpickle(trainfile3)
trainlabelsl3=traindict[b'labels']
trainlabels3=np.array(trainlabelsl3)
traindata3=traindict3[b'data']
traindata3 = traindata3/255

trainfile4 = "data/data_batch_4"
traindict4=unpickle(trainfile4)
trainlabelsl4=traindict[b'labels']
trainlabels4=np.array(trainlabelsl4)
traindata4=traindict4[b'data']
traindata4 = traindata4/255

trainfile5 = "data/data_batch_5"
traindict5=unpickle(trainfile5)
trainlabelsl3=traindict[b'labels']
trainlabels3=np.array(trainlabelsl3)
traindata3=traindict3[b'data']
traindata3 = traindata3/255
'''

traindata=np.concatenate((traindata1, traindata2))
trainlabels=np.concatenate((trainlabels1,trainlabels2))


testfile = "data/test_batch"
testdict=unpickle(testfile)
testlabelsl=testdict[b'labels']
testlabels=np.array(testlabelsl)
testdata=testdict[b'data']
testdata = testdata/255

#STOLEN FROM EXERCISES
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
    #print(f'Trn shape: {trn_set.shape}')
    #Zero arrays for storing means and covs.
    means = np.zeros((10,d))
    covs = np.zeros((10, d, d))
    
    #For each class compute mean and cov.
    for i in range(10):
        indx = trn_targets == i
        #print(f'index: {indx}')
        means[i] = np.mean(trn_set[indx], axis = 0)
        #print(f'Means of: {i} :{means[i]}')
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
#END OF STOLEN CODE
'''
#splitter
trn = [None] * 10
for b in range (10):
    trn[b]=np.array([np.zeros(3072)])
    for x in range(len(traindict[b'labels'])):
        if trainlabels[x]==b:
            trn[b]=np.concatenate((trn[b], np.array([traindata[x]])), axis=0)
    trn[b]=np.delete(trn[b], 0, 0)

#print(trn[0])
'''
n_components = 100

#Start of stolen code again 
#print(traindata)
#PCA
pca = PCA(n_components = n_components)
trn_pca_set = pca.fit_transform(traindata)
tst_pca_set = pca.transform(testdata)
'''
#LDA
lda = LDA(n_components = n_components)
trn_lda_set = lda.fit_transform(traindata, trainlabels)
tst_lda_set = lda.transform(testdata)
'''
#Proportion of Variance
pov_pca = np.sum(pca.explained_variance_ratio_)
#pov_lda = np.sum(lda.explained_variance_ratio_)

#%%Classification
#Compute the parameters for each PCA and LDA reduced data.
pca_means, pca_covs = est_params(trn_pca_set, trainlabels)
#lda_means, lda_covs = est_params(trn_lda_set, trainlabels)

#Compute predictions
pca_pred = predict(tst_pca_set, pca_means, pca_covs)
#lda_pred = predict(tst_lda_set, lda_means, lda_covs)

#Compute accuracy
pca_acc = np.sum(pca_pred == testlabels)/len(testlabels) * 100
#lda_acc = np.sum(lda_pred == testlabels)/len(testlabels) * 100
print("PCA Accuracy: {:.2f}".format(pca_acc))
#print("LDA Accuracy: {:.2f}".format(lda_acc))


                     