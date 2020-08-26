from keras.engine.saving import load_model
from scipy import signal
from scipy.io.wavfile import read
import tensorflow as tf
import numpy as np
import numpy
import pickle
from numpy import  newaxis
import matplotlib.pyplot as plt
from collections import Counter


from tensorflow.keras.layers import Input, Dense, Conv1D, Activation, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential


#######################
instruClass=10 #Generat wavefroms by giving the class code here.

filenames = ["/home/fac/cmh/tfrecord/nsynth-test.tfrecord"]
filenames = ["/home/fac/cmh/tfrecord/nsynth-test.tfrecord"]
raw_dataset = tf.data.TFRecordDataset(filenames)

X=[]
y=[]
instru_scr=[]
pitch=[]
velocity=[]
qualities=[]
#names=[]
counter=0 #
for raw_record in raw_dataset.take(4096):#4096

     example = tf.train.Example()
     example.ParseFromString(raw_record.numpy())
     if(example.features.feature["instrument_family"].int64_list.value[0]!=9):
          X.append(example.features.feature["audio"].float_list.value)
          y.append(example.features.feature["instrument_family"].int64_list.value[0])
          instru_scr.append(example.features.feature["instrument_source"].int64_list.value[0])
          pitch.append(example.features.feature["pitch"].int64_list.value[0])
          velocity.append(example.features.feature["velocity"].int64_list.value[0])
          qualities.append(example.features.feature["qualities"].int64_list.value)
          counter=counter+1
          
          #break

yClass=[]
for i in range(0,len(y)):
    temp=[0] * 11
    temp[y[i]]=1
    yClass.append(temp)
Xnew=[]
counter=0
for i in  X:
    f = signal.resample(i, 8000)
    i = None
    f = numpy.append(f, pitch[counter])
    f = numpy.append(f, velocity[counter])
    f = numpy.append(f, instru_scr[counter])
    f = numpy.append(f, qualities[counter])
    pitch[counter] = None
    velocity[counter] = None
    instru_scr[counter] = None
    qualities[counter] = None

    
    Xnew.append(f)
    counter = 1 + counter
Xtest= np.array(Xnew)

ytest= np.array(yClass)


 
b = Xtest[:, :, newaxis]
Xtest=b

model = tf.keras.models.load_model('model2.h5') #Model name here.
pred= model.predict(Xtest,verbose=1)


#########

#########

maxVal=0
minVal=10000
maxIndex=0
minIndex=0
instruClass=0
for i in range(0,len(pred)):
    if(pred[i][instruClass]<minVal):
        minVal=pred[i][instruClass]
        minIndex=i
    if (pred[i][instruClass] > maxVal):
        maxVal = pred[i][instruClass]
        maxIndex = i


audioFile="/nsynth-test/"+"/audio/"+names[maxIndex]+".wav" #

audio=read(audioFile)
sh = np.array(audio[1], dtype=float).shape[0]
length = sh / audio[0]
timen = np.linspace(0., length, sh)
fig=plt.plot(timen, audio[1])
plt.title('Class:'+instFamlityStr[maxIndex]+' ; Correct class probability: high')
plt.xlabel("Time [s] ")
plt.ylabel("Amplitude")
plt.show();
#plt.savefig(instFamlityStr[maxIndex]+'HP.png')


audioFile="/nsynth-test/"+"/audio/"+names[minIndex]+".wav"
#sh= np.array(X[maxIndex],dtype=float).shape[0]
#length=sh/audio[0]
audio=read(audioFile)
sh = np.array(audio[1], dtype=float).shape[0]
length = sh / audio[0]
timen = np.linspace(0., length, sh)
fig=plt.plot(timen, audio[1])
plt.title('Class:'+instFamlityStr[minIndex]+' ; Correct class probability: low')
plt.xlabel("Time [s]  ")
plt.ylabel("Amplitude")
plt.show();
#plt.savefig(instFamlityStr[maxIndex]+'LP.png')










