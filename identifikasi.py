import librosa
import librosa.display
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from random import randint

path=["Bu Gini","Bu Yam","Bu Yayuk","Busia","Jainiah","Vidiya"] 
# Model reconstruction from JSON file
with open('model_architecture 1000 180 90 no noise.json', 'r') as f:
    model = tf.keras.models.model_from_json(f.read())

# Load weights into the new model
model.load_weights('model_weights 1000 180 90 no noise.h5')
#i=1
#a=0
#for a in range(len(path)):
#    i=1
#    while i<=50:
#        data = path[a]+"/"+str(i)+".wav"
#        print(data)
#        y, sr = librosa.load(data,offset=0,duration=3, sr=44100)
#        B=librosa.feature.mfcc(y=y, sr=sr)
#        B=(B-np.min(B))/(np.max(B)-np.min(B))
#        data=B.flatten()
#        data.shape=(1,5180)

#        predictRate=model.predict(data)
#        prediksi=model.predict_classes(data)
#        indeks=prediksi[0]
#        print("nilai prediksi semua class= ",predictRate)
#        print("nilai prediksi yang dipilih= ",indeks)
#        print("Seharusnya=", str(path[a]))
#        i=i+1
#    a=a+1
history_dict = model.history
print(history_dict.keys())
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(len(acc))
plt.figure(num='1000 180 90 no noise');
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()
plt.show()
#data = "Bu Gini/32.wav"
#y, sr = librosa.load(data,offset=0,duration=3,mono=True, sr=44100)
#B=librosa.feature.mfcc(y=y, sr=44100)
#B=(B-np.min(B))/(np.max(B)-np.min(B))
#data=B.flatten()
#data.shape=(1,5180)
#print(data)
#predictRate=model.predict(data)
#prediksi=model.predict_classes(data)
#indeks=prediksi[0]
#print("nilai prediksi semua class= ",predictRate)
#print("nilai prediksi yang dipilih= ",indeks)    

