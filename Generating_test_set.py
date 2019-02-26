
# Extracting features from the audio files
import pandas as pd
import soundfile as sf
import librosa 
import glob
import numpy as np


test_set=pd.read_csv('Test_set.csv')
print(test_set)
audiofiles=list(test_set.iloc[:,0])
dt=test_set.set_index(['performance_id'])['gender'].to_dict()

count=0
features=[]
for key in dt:
    
    filename='/users/saniyaambavanekar/Documents/MLSP_Project/Test_set/'+key+'.wav'
    
    clip,sr=librosa.load(filename)
    
    gender_target=dt.get(key)
    
    mat_stft = np.abs(librosa.core.stft(y = clip, n_fft = 1024,hop_length= 512))
    
    mat_mfcc = np.mean(librosa.feature.mfcc(clip,sr,n_mfcc = 40),axis = 1)

    mat_chroma = np.mean(librosa.feature.chroma_stft(S=mat_stft, sr=sr),axis=1)

    mat_mel = np.mean(librosa.feature.melspectrogram(clip, sr=sr),axis=1)

    mat_contrast = np.mean(librosa.feature.spectral_contrast(S=mat_stft, sr=sr),axis=1)

    mat_tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(clip),
    sr=sr),axis=1)
    
    zero_cross = np.mean(librosa.feature.zero_crossing_rate(clip))
    cent = np.mean(librosa.feature.spectral_centroid(clip,sr))
    
    if(count>1000):
        break
    count=count+1
    ext_features = np.hstack([key,mat_mfcc,mat_chroma,mat_mel, mat_contrast,mat_tonnetz,zero_cross,cent,gender_target])
    #print(ext_features)
    features.append(ext_features)
    print(count)
    
    
df=pd.DataFrame(features)

df.to_csv('test_gender.csv', sep=',')

