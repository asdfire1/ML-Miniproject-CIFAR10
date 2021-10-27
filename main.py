
import numpy as np
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
trainfile = "data/data_batch_1"
traindict=unpickle(trainfile)
testfile = "data/test_batch"
testdict=unpickle(testfile)
trainlabels=traindict[b'labels']
traindata=traindict[b'data']

#splitter
trn = [None] * 10
for b in range (10):
    trn[b]=np.array([np.zeros(3072)])
    for x in range(len(traindict[b'labels'])):
        if trainlabels[x]==b:
            trn[b]=np.concatenate((trn[b], np.array([traindata[x]])), axis=0)
    trn[b]=np.delete(trn[b], 0, 0)

print(trn[0])
    