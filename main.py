import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
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

#Code from exercises
n_components=190
pca = PCA(n_components = n_components)
trn_pca_set = pca.fit_transform(traindata)
tst_pca_set = pca.transform(testdata)
#A list of the class names.
classes = np.arange(10)

#%%Support Vector Machine (SVM)
#Maximum number of iterations. Even if the SVM doesn't converge it will stop anyway.
n_iter = 3000
#Create SVM for classification (SVC). The standard kernel in the scikit-learn
#is the RBF (Gaussian Kernel).
svc = svm.SVC(max_iter = n_iter)

start_time = time.time()    #Time in seconds when starting the training/fiting.
svc.fit(trn_pca_set, trainlabels)   #Train/fit the SVC
fit_time = time.time() - start_time     #Compute the training/fiting time in seconds
print("Training Time: {:2.0f}m{:2.0f}s".format(fit_time//60, fit_time%60))

#Predictions
pred = svc.predict(tst_pca_set)

end_time = time.time() - start_time #Compute the total train and predict time in seconds
print("Total Running Time: {:2.0f}m{:2.0f}s".format(end_time//60, end_time%60))

#Compute accuracy
acc = np.sum(pred == testlabels)/len(testlabels) * 100
print("SVM Accuracy: {:.2f}".format(acc))