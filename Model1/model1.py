from keras.engine.saving import load_model
from scipy import signal
import tensorflow as tf
import numpy as np
import pickle
from numpy import  newaxis


from tensorflow.keras.layers import Input, Dense, Conv1D, Activation, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#######################


filenames = ["~/tfrecord/nsynth-valid.tfrecord"]
raw_dataset = tf.data.TFRecordDataset(filenames)

X=[]
y=[]
counts = dict()

counter=0 #
for raw_record in raw_dataset.take(12678):#12678
     example = tf.train.Example()
     example.ParseFromString(raw_record.numpy())
     if(example.features.feature["instrument_family"].int64_list.value[0]!=9):
          temp = example.features.feature["instrument_str"].bytes_list.value[0]
          temp = temp[:-4]
          if(counts.get(temp)==None or counts.get(temp)<221):
            X.append(example.features.feature["audio"].float_list.value)
            y.append(example.features.feature["instrument_family"].int64_list.value[0])
            counts[temp] = counts.get(temp, 0) + 1
            counter=counter+1
            
          #break

yClass=[]
counts=None

for i in range(0,len(y)):
    temp=[0] * 11
    temp[y[i]]=1
    yClass.append(temp)

Xnew=[]

counter=0
for i in  X:
    #print (i)
    f = signal.resample(i, 8000)
    counter=1+counter

    Xnew.append(f)
XValidate= np.array(Xnew)

yValidate= np.array(yClass)


#######################
#######################


filenames = ["~/tfrecord/nsynth-test.tfrecord"]
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

#######################


filenames = ["~/tfrecord/nsynth-train.tfrecord"]
raw_dataset = tf.data.TFRecordDataset(filenames)
X=[]
y=[]
counts = dict()
#names=[]
counter=0
for raw_record in raw_dataset.take(283704):
     example = tf.train.Example()
     example.ParseFromString(raw_record.numpy())
     if(example.features.feature["instrument_family"].int64_list.value[0]!=9):
          temp = example.features.feature["instrument_str"].bytes_list.value[0]
          temp = temp[:-4]
          if(counts.get(temp)==None or counts.get(temp)<3000):
            X.append(example.features.feature["audio"].float_list.value)
            y.append(example.features.feature["instrument_family"].int64_list.value[0])
            counts[temp] = counts.get(temp, 0) + 1
            counter=counter+1
            
          #break

yClass=[]
counts=None
for i in range(0,len(y)):
    temp=[0] * 11
    temp[y[i]]=1
    yClass.append(temp)
yTrain=np.array(yClass)
Xnew=[]
counter=0
for i in  X:
    f = signal.resample(i, 8000)
    counter=1+counter
    Xnew.append(f)
XTrain= np.array(Xnew)


b = Xtest[:, :, newaxis]
Xtest=b
b = XValidate[:, :, newaxis]
XValidate=b
b = XTrain[:, :, newaxis]
XTrain=b
model = Sequential()

model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(8000,1)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(11, activation='softmax'))

# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
history=model.fit(XTrain, yTrain, validation_data=(XValidate,yValidate),epochs=70, batch_size=100)


_, accuracy = model.evaluate(Xtest, ytest)

print('Accuracy: %.2f' % (accuracy*100))
import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()










