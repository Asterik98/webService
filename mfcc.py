import librosa
import librosa.display
import numpy as np
import glob
import os
import pandas as pd
import sounddevice as sd
import matplotlib.pyplot as plt
import shutil
from sklearn.preprocessing import normalize
path='/KULIAH/Tugas Akhir/Code'
noisePath='/KULIAH/Tugas Akhir/Code/Noise'
names=[d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
i=0
a=0

for a in range(len(names)):
    if names[a]=="Data MFCC":
        shutil.rmtree(path+'/Data MFCC')
        del names[a]
    if a==len(names)-1:
        break
print(names)    
for i in range(len(names)):
    if names[i]!="Noise":
        name=names[i]
        newPath=path+'/Data MFCC/'+name+' MFCC'
        try:
            os.makedirs(newPath)
        except OSError:
            print ("Creation of the directory %s failed" % newPath)
        else:
            print ("Successfully created the directory %s " % newPath)
        b=0
        for filename in glob.glob(os.path.join(path+'/'+name, '*.wav')):
            noiseFile=noisePath+"/"+str(b+1)+'.wav'
            #head, tail = os.path.split(filename)
            #print(head)
            #plt.subplot(2,1,1)
            y, sr = librosa.load(filename,offset=0,duration=3, sr=44100)
            #librosa.display.waveplot(y, sr=sr)
            #plt.subplot(2,1,2)
            y2, sr2 = librosa.load(noiseFile,offset=0,duration=3,mono=True, sr=44100)
            #y=y+y2
            #librosa.display.waveplot(y, sr=sr)
            #plt.show()
            #librosa.output.write_wav(newPath+'/'+tail, y, sr, norm=False)
            B=librosa.feature.mfcc(y=y, sr=sr)
            B=(B-np.min(B))/(np.max(B)-np.min(B))
            #B=normalize(B, axis=1, norm='l1')
            if b==0:
                data=B.flatten()
            else:
                data=np.vstack((data,B.flatten()))
            b=b+1
            if b==50:
                pd.DataFrame(data).to_csv(newPath+'/'+name+' MFCC.csv',index=False) 
    
    

print("Done")
#tambah gaussian white noise
