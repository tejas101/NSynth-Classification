from keras.engine.saving import load_model
from scipy import signal
import tensorflow as tf
import numpy as np
import pickle
from numpy import  newaxis
from collections import Counter
from sklearn.metrics import  confusion_matrix

import keras
#from tensorflow.keras.layers import Input, Dense, Conv1D, Activation, Dropout, MaxPooling1D, Flatten
#from tensorflow.keras.models import Sequential

filenames = ["/home/fac/cmh/tfrecord/nsynth-test.tfrecord"]
raw_dataset = tf.data.TFRecordDataset(filenames)

X=[]
y=[]

counter=0 #
for raw_record in raw_dataset.take(4096):#4096
     example = tf.train.Example()
     example.ParseFromString(raw_record.numpy())
     if(example.features.feature["instrument_family"].int64_list.value[0]!=9):
          X.append(example.features.feature["audio"].float_list.value)
          y.append(example.features.feature["instrument_family"].int64_list.value[0])

          counter=counter+1
          
          

yClass=[]
for i in range(0,len(y)):
    temp=[0] * 11
    temp[y[i]]=1
    yClass.append(temp)

Xnew=[]
counter=0
for i in  X:
    
    f = signal.resample(i, 8000)
    counter=1+counter
    Xnew.append(f)
Xtest= np.array(Xnew)

ytest= np.array(yClass)
# evaluate the keras model

model = tf.keras.models.load_model('model1.h5')# Model Name

#Xtest=pickle.load(open(""))#Load picked or the processed Xtest
#ytest=pickle.load(open(""))#Load picked or the processed ytest
b = Xtest[:, :, newaxis]
Xtest=b

pred= model.predict(Xtest,batch_size=100,verbose=1)
pred = (pred > 0.5)
ytest = (ytest > 0.5)
pred=np.argmax(pred, axis=1)
ytest=np.argmax(ytest, axis=1)

cm= confusion_matrix(ytest,pred)

history=[[],[],[],[]]

#Code courtesy  : https://www.kaggle.com/grfiv4/plot-a-confusion-matrix

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()



plot_confusion_matrix(cm, normalize = True,target_names = ['bass', 'brass', 'flute','guitar', 'keyboard', 'mallet','organ', 'reed', 'string','vocal'],title= "Confusion Matrix, Normalized")



