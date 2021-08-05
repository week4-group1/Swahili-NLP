import os
import librosa
import librosa.feature
import numpy as np

def gen_features(audio_files_directory, out_path):
    sr = 44100
    for root, directories, files in os.walk(audio_files_directory):
        for filename in files:
            audio_path = os.path.join(root, filename)
            audio_out = os.path.join(out_path, filename)
            audio_data, ssr = librosa.load(audio_path, sr=44100) # getting y
            spectogram = librosa.stft(abs(audio_data))
            mfcc_feature_list = librosa.feature.mfcc(spectogram,sr=44100) # create mfcc features
            np.savetxt(str(audio_out), spectogram, delimiter ="\t")
                                      
a = "C:\\Users\\cthru\Documents\\psnl_projects\\10academy\\Swahili-NLP\\data\\data\\train\\wav"
b = "C:\\Users\\cthru\\Documents\\psnl_projects\\10academy\\Swahili-NLP\\data\\features"
        
gen_features(a,b) 