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
traindata1 = traindata1/255

trainfile2 = "data/data_batch_2"
traindict2=unpickle(trainfile2)
trainlabelsl2=traindict[b'labels']
trainlabels2=np.array(trainlabelsl2)
traindata2=traindict2[b'data']
traindata2 = traindata2/255

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
trainlabelsl5=traindict[b'labels']
trainlabels5=np.array(trainlabelsl5)
traindata5=traindict5[b'data']
traindata5 = traindata5/255

#traindata=traindata1
#trainlabels=trainlabels1
traindata_all=np.concatenate((traindata1, traindata2, traindata3, traindata4, traindata5))
trainlabels_all=np.concatenate((trainlabels1,trainlabels2, trainlabels3, trainlabels4, trainlabels5))
#Splitting the data into training and validation set
validdata, traindata = traindata_all[:5000], traindata_all[5000:]
validlabels, trainlabels = trainlabels_all[:5000], trainlabels_all[5000:]

testfile = "data/test_batch" #Path to testing data file
testdict=unpickle(testfile)
testlabelsl=testdict[b'labels']
testlabels=np.array(testlabelsl)
testdata=testdict[b'data']
testdata = testdata/255
#neural network structure
model = keras.models.Sequential([ 
                                 keras.layers.Dense(300, activation = "relu" ), 
                                 keras.layers.Dense(100, activation = "relu" ),
                                 keras.layers.Dense(100, activation = "relu" ), 
                                 keras.layers.Dense(100, activation = "relu" ),
                                 keras.layers.Dense(10, activation = "softmax" )]) 
#model.summary()  #Show the structure of the neural network
model.compile(loss = "sparse_categorical_crossentropy",  #loss function
              optimizer = "sgd",                         #optimizer against the loss function - stochastic gradient descent
              metrics = ["accuracy"])                    #metric to use in addition to loss

#training the neural network
history = model.fit(traindata, 
                    trainlabels, 
                    epochs = 50, 
                    validation_data = (validdata, validlabels))
#plots
pd.DataFrame(history.history).plot(figsize = (16, 10)) 
plt.grid(True) 
plt.gca().set_ylim(0, 1) 
plt.show() 


#Computing the accuracy of predictions
#pca_acc = np.sum(nnpredict == testlabels)/len(testlabels) * 100
model.evaluate(testdata,testlabels)
end_time = time.time() - start_time     #Compute the training/fiting time in seconds
print("Running time: {:2.0f}m{:2.0f}s".format(end_time//60, end_time%60))
#print("PCA Accuracy: {:.2f}".format(pca_acc))



                     