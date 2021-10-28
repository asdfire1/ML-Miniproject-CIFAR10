import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


trainfile = "data/data_batch_1"
traindict=unpickle(trainfile)
trainlabels=traindict[b'labels']
traindata=traindict[b'data']
traindata = traindata/255

testfile = "data/test_batch"
testdict=unpickle(testfile)
testlabels=testdict[b'labels']
testdata=testdict[b'data']
testdata = testdata/255


#splitter
trn = [None] * 10
for b in range (10):
    trn[b]=np.array([np.zeros(3072)])
    for x in range(len(traindict[b'labels'])):
        if trainlabels[x]==b:
            trn[b]=np.concatenate((trn[b], np.array([traindata[x]])), axis=0)
    trn[b]=np.delete(trn[b], 0, 0)

#print(trn[0])
n_components = 9

#PCA
pca = PCA(n_components = n_components)
trn_pca_set = pca.fit_transform(traindata)
tst_pca_set = pca.transform(testdata)

#LDA
lda = LDA(n_components = n_components)
trn_lda_set = lda.fit_transform(traindata, trainlabels)
tst_lda_set = lda.transform(testdata)
