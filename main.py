#Libraries, imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import time
#Function for unpacking the data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


start_time = time.time()
#Unpacking the data files, normalizing them and saving them as variables


trainfile = "data/data_batch_1"
traindict=unpickle(trainfile)
trainlabelsl=traindict[b'labels']
trainlabels1=np.array(trainlabelsl)
traindata1=traindict[b'data']
del traindict
del trainlabelsl
#traindata1 = traindata1/255

trainfile2 = "data/data_batch_2"
traindict2=unpickle(trainfile2)
trainlabelsl2=traindict2[b'labels']
trainlabels2=np.array(trainlabelsl2)
traindata2=traindict2[b'data']
del traindict2
del trainlabelsl2
#traindata2 = traindata2/255

trainfile3 = "data/data_batch_3"
traindict3=unpickle(trainfile3)
trainlabelsl3=traindict3[b'labels']
trainlabels3=np.array(trainlabelsl3)
traindata3=traindict3[b'data']
del traindict3
del trainlabelsl3
#traindata3 = traindata3/255

trainfile4 = "data/data_batch_4"
traindict4=unpickle(trainfile4)
trainlabelsl4=traindict4[b'labels']
trainlabels4=np.array(trainlabelsl4)
traindata4=traindict4[b'data']
del traindict4
del trainlabelsl4
#traindata4 = traindata4/255

trainfile5 = "data/data_batch_5"
traindict5=unpickle(trainfile5)
trainlabelsl5=traindict5[b'labels']
trainlabels5=np.array(trainlabelsl5)
traindata5=traindict5[b'data']
del traindict5
del trainlabelsl5
#traindata5 = traindata5/255

#traindata=traindata1
#trainlabels=trainlabels1
traindata_all=np.concatenate((traindata1, traindata2, traindata3, traindata4, traindata5))
del traindata1, traindata2, traindata3, traindata4, traindata5
trainlabels_all=np.concatenate((trainlabels1,trainlabels2, trainlabels3, trainlabels4, trainlabels5))
del trainlabels1,trainlabels2, trainlabels3, trainlabels4, trainlabels5
traindata1r=np.reshape(traindata_all, (50000,32,32,3), order='F')
del traindata_all

testfile = "data/test_batch" #Path to testing data file
testdict=unpickle(testfile)
testlabelsl=testdict[b'labels']
testlabels=np.array(testlabelsl)
testdata=testdict[b'data']
del testdict
del testlabelsl
#testdata = testdata/255
testdatar=np.reshape(testdata, (10000,32,32,3), order='F')
del testdata


#adapting names to match the sample code
y_train_full=trainlabels_all
X_train_full=traindata1r
X_test=testdatar
y_test=testlabels
#deleting  old variables
del trainlabels_all
del traindata1r
del testdatar
del testlabels

#sample code
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

#normalizing data sn building a test set

X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

#X_train = X_train[..., np.newaxis]
#X_valid = X_valid[..., np.newaxis]
#X_test = X_test[..., np.newaxis]
#neural network structure
model = keras.models.Sequential([keras.layers.InputLayer(input_shape=(32, 32, 3)),
                                 keras.layers.Conv2D(64, 8, activation="relu", padding= "same", input_shape=[32, 32, 3]),
                                 keras.layers.MaxPooling2D(2),
                                 keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
                                 keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
                                 keras.layers.MaxPooling2D(2),
                                 keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
                                 keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
                                 keras.layers.MaxPooling2D(2),
                                 keras.layers.Flatten(),
                                 keras.layers.Dense(128, activation="relu"),
                                 keras.layers.Dropout(0.5),
                                 keras.layers.Dense(64, activation="relu"),
                                 keras.layers.Dropout(0.5),
                                 keras.layers.Dense(10, activation="softmax")
                                 ])
#model.summary()  #Show the structure of the neural network
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

#training the neural network
history = model.fit(X_train, 
                    y_train, 
                    epochs = 30, 
                    validation_data = (X_valid, y_valid))
#plots
pd.DataFrame(history.history).plot(figsize = (16, 10)) 
plt.grid(True) 
plt.gca().set_ylim(0, 1) 
plt.show() 


#Computing the accuracy of predictions
#pca_acc = np.sum(nnpredict == testlabels)/len(testlabels) * 100
model.evaluate(X_test,y_test)
end_time = time.time() - start_time     #Compute the training/fiting time in seconds
print("Running time: {:2.0f}m{:2.0f}s".format(end_time//60, end_time%60))
#print("PCA Accuracy: {:.2f}".format(pca_acc))



                     