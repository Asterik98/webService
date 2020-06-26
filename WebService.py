from flask import Flask,request,jsonify
import base64
import scipy.io.wavfile
import os
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import librosa
import librosa.display
import glob
import pandas as pd
import sounddevice as sd
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
app = Flask(__name__)
@app.route('/', methods=['POST'])
def presensi():
    tic = time.process_time()
    path='/home/ubuntu/webService'
    names=[d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    i=0
    while i<len(names):
        if names[i]=="Data MFCC" or names[i]=="Noise" or names[i]==".git":
            names.remove(names[i])
        i=i+1
    req_data = base64.b64decode(request.json['audio'])
    filePath="test.wav"
    if os.path.exists(filePath):
        os.remove(filePath)
    else:
        print("Can not delete the file as it doesn't exists")
    with open('test.wav', mode='bx') as f:
        f.write(req_data)
    y, sr = librosa.load('test.wav',offset=0,duration=3, sr=44100)
    B=librosa.feature.mfcc(y=y, sr=sr)
    B=(B-np.min(B))/(np.max(B)-np.min(B))
    data=B.flatten()
    data.shape=(1,5180)
    with open('model_architecture.json', 'r') as f:
        model = tf.keras.models.model_from_json(f.read())
    model.load_weights('model_weights.h5')
    prediksi=model.predict_classes(data).tolist()
    print(prediksi[0])
    print(names[prediksi[0]])
    toc = time.process_time()
    print("Waktu:",toc-tic)
    return jsonify({'prediction': names[prediksi[0]]})
   
@app.route('/new', methods=['POST'])
def save_data():
    req_data = request.json['audio']
    nama=request.json['name']
    nama=nama[0]
    try:
         os.makedirs(nama)
    except OSError:
            print ("Creation of the directory %s failed" % nama)
    else:
            print ("Successfully created the directory %s " % nama)
    i=0
    for i in range(len(req_data)):
        with open(nama+'/'+str(i+1)+'.wav', mode='wb') as f:
            f.write(base64.b64decode(req_data[i]))
    mfcc(nama)
    ann()
    return jsonify({'hasil': 'Telah Tersimpan'})

def mfcc(nama):
    newPath='Data MFCC/'+nama+' MFCC'
    noisePath='Noise'
    try:
        os.makedirs(newPath)
    except OSError:
        print ("Creation of the directory %s failed" % newPath)
    else:
        print ("Successfully created the directory %s " % newPath)
    b=0
    for filename in glob.glob(os.path.join(nama, '*.wav')):
        noiseFile=noisePath+"/"+str(b+1)+'.wav'
        y, sr = librosa.load(filename,offset=0,duration=3, sr=44100)
        y2, sr2 = librosa.load(noiseFile,offset=0,duration=3, sr=44100)
        if b<=29 && b>=44:
            y=y+y2
        B=librosa.feature.mfcc(y=y, sr=sr)
        B =  (B - np.min(B)) / (np.max(B) - np.min(B))
        if b==0:
            data=B.flatten()
        else:
            data=np.vstack((data,B.flatten()))
        b=b+1
        if b==50:
            pd.DataFrame(data).to_csv(newPath+'/'+nama+' MFCC.csv',index=False)

def ann():
    path='Data MFCC'
    names=[d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    i=0
    for i in range(len(names)):
        dataset = pd.read_csv(path+"/"+names[i]+"/"+names[i]+".csv")
        label=np.full(50,i)
        if i==0:
            X_train,X_test,Y_train,Y_test = train_test_split(dataset,label,test_size = 0.2,shuffle=False, stratify = None)
        else:
            X_trainAdd,X_testAdd,Y_trainAdd,Y_testAdd = train_test_split(dataset,label,test_size = 0.2)
            X_train=np.vstack((X_train,X_trainAdd))
            X_test=np.vstack((X_test,X_testAdd))
            Y_train=np.append(Y_train,Y_trainAdd)
            Y_test=np.append(Y_test,Y_testAdd)
        i=i+1
       

    model = tf.keras.Sequential([
            tf.keras.layers.Dense(1000, input_dim=X_train.shape[1], activation='relu'),
            tf.keras.layers.Dense(180, activation='relu'),
            tf.keras.layers.Dense(90, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(i, activation='softmax')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=1e-6)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(X_train, Y_train,validation_data=(X_test, Y_test), epochs=1000, batch_size=40,verbose=2)
    akurasiTrain=model.evaluate(X_train, Y_train)
    print("Akurasi Train= "+str(akurasiTrain[1]*100)+"%")
    prediksi = model.predict_classes(X_test)

    benar=0
    print("Akurasi : ",accuracy_score(Y_test, prediksi)*100,"%")
        
    model.save_weights('model_weights.h5')
    with open('model_architecture.json', 'w') as f:
        f.write(model.to_json())
    print("selesai")
    
@app.route('/hapus', methods=['POST'])
def delete_data():
    nama = request.json['data']
    i=0
    while i<len(nama):
        filepath=str(nama[i])
        shutil.rmtree(filepath)
        shutil.rmtree('Data MFCC/'+filepath+' MFCC')
        i=i+1
    nama=None
    ann()
    return jsonify({'hasil': 'Telah Dihapus'})

app.run(host='0.0.0.0', port=5000, debug=True)

