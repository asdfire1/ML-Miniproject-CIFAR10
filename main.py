import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from sklearn import svm
import time

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


trainfile = "data/data_batch_1"
traindict=unpickle(trainfile)
trainlabelsl=traindict[b'labels']
trainlabels=np.array(trainlabelsl)
traindata=traindict[b'data']
traindata = traindata/255

testfile = "data/test_batch"
testdict=unpickle(testfile)
testlabelsl=testdict[b'labels']
testlabels=np.array(testlabelsl)
testdata=testdict[b'data']
testdata = testdata/255

#STOLEN FROM EXERCISES
#A list of the class names.
classes = np.arange(10)

#%%Support Vector Machine (SVM)
#Maximum number of iterations. Even if the SVM doesn't converge it will stop anyway.
n_iter = 3000

<<<<<<< Updated upstream
    Returns
    -------
    means : numpy.ndarray
        Mean vectors for each class.
    covs : numpy.ndarray
        Covariance matrices for each class.
    '''
    #Get the data dimension
    _, d = trn_set.shape 
    print(f'Trn shape: {trn_set.shape}')
    #Zero arrays for storing means and covs.
    means = np.zeros((10,d))
    covs = np.zeros((10, d, d))
    
    #For each class compute mean and cov.
    for i in range(10):
        indx = trn_targets == i
        print(f'index: {indx}')
        means[i] = np.mean(trn_set[indx], axis = 0)
        print(f'Means of: {i} :{means[i]}')
        covs[i] = np.cov(trn_set[indx].T)
    return means, covs
=======
#Create SVM for classification (SVC). The standard kernel in the scikit-learn
#is the RBF (Gaussian Kernel).
svc = svm.SVC(max_iter = n_iter)
>>>>>>> Stashed changes

start_time = time.time()    #Time in seconds when starting the training/fiting.
svc.fit(traindata, trainlabels)   #Train/fit the SVC
fit_time = time.time() - start_time     #Compute the training/fiting time in seconds
print("Training Time: {:2.0f}m{:2.0f}s".format(fit_time//60, fit_time%60))

#Predictions
pred = svc.predict(testdata)

<<<<<<< Updated upstream
#print(trn[0])
'''
n_components = 100

#Start of stolen code again 
print(traindata)
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

=======
end_time = time.time() - start_time #Compute the total train and predict time in seconds
print("Total Running Time: {:2.0f}m{:2.0f}s".format(end_time//60, end_time%60))

#Compute accuracy
acc = np.sum(pred == testlabels)/len(testlabels) * 100
print("SVM Accuracy: {:.2f}".format(acc))

#%%Confusion matrix
#Compute the confusion matrix
cm = confusion_matrix(testlabels, pred, normalize = "true")

#Prepare for plotting
cm = ConfusionMatrixDisplay(cm, classes)

#Plot Confusion matrices
fig, ax = plt.subplots()
cm.plot(cmap = "Blues", ax = ax)
ax.set_title("Confusion Matrix SVM")
>>>>>>> Stashed changes

                     