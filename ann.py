import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

#dataset import
path='Data MFCC'
names=[d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
# Neural network
i=0
for i in range(len(names)):
    dataset = pd.read_csv(path+"/"+names[i]+"/"+names[i]+".csv")
    label=np.full(50,i)
    if i==0:
        X_train,X_test,Y_train,Y_test = train_test_split(dataset,label,test_size = 0.9,shuffle=False, stratify = None)
    else:
        X_trainAdd,X_testAdd,Y_trainAdd,Y_testAdd = train_test_split(dataset,label,test_size = 0.9,shuffle=False, stratify = None)
        X_train=np.vstack((X_train,X_trainAdd))
        X_test=np.vstack((X_test,X_testAdd))
        Y_train=np.append(Y_train,Y_trainAdd)
        Y_test=np.append(Y_test,Y_testAdd)
    i=i+1
print(Y_train)   
akurasi=0
model = tf.keras.Sequential([
        tf.keras.layers.Dense(1000, input_dim=X_train.shape[1], activation='relu'),
        tf.keras.layers.Dense(180, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(i, activation='softmax')
        ])
opt = tf.keras.optimizers.Adam(learning_rate=1e-6)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(X_train, Y_train,validation_data=(X_test, Y_test), epochs=1000, batch_size=40,verbose=2)
akurasiTrain=model.evaluate(X_train, Y_train)
print("Akurasi Train= "+str(akurasiTrain[1]*100)+"%")
prediksi = model.predict_classes(X_test)
print(prediksi)
benar=0
print("Akurasi : ",accuracy_score(Y_test, prediksi)*100,"%")
print(classification_report(Y_test, prediksi))
print(pd.DataFrame(
    confusion_matrix(Y_test, prediksi), 
    index=['Act:0', 'Act:1','Act:2','Act:3','Act:4','Act:5'], 
    columns=['Pred:0', 'Pred:1','Pred:2','Pred:3','Pred:4','Pred:5']))  
model.save_weights('model_weights 1000 180 50 coba data 5.h5')
with open('model_architecture 1000 180 50 coba data 5.json', 'w') as f:
    f.write(model.to_json())
print("1000 180 50 coba data 5")
print("selesai")    
    
history_dict = history.history
print(history_dict.keys())
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(len(acc))
plt.figure(num='1000 180 50 coba data 5');
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()
plt.show()

