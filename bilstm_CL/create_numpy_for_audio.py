
import subprocess as sub
import sys
import os
import numpy as np
from numpy import genfromtxt

def create_numpy_for_audio(audio_file, featureplan):
    """This function is based on YAAFE. It will return 2D Array which is features of audio file. 
    Also it will save the numpy array. You can use this function for one file.
    
    Its arguments:
    audio_file: Path of audio file, it can be wav, mp3, ogg etc.
    featureplan: Text file which introduce which features will be extracted.
        Available options: 
            - pyannote_based
            - mfcc
    """
    
    if (featureplan=="pyannote_based.txt"):
        cmd = "yaafe -c " + featureplan + " -r 16000 "  + audio_file + " -p Precision=6 -p Metadata=False -n"
        print (cmd)
	
        os.system(cmd)
        filename = (audio_file.split("/")[-1]).split(".")[0]

        my_data = genfromtxt(audio_file + ".mfcc.csv", delimiter=',')
        my_data = np.append(my_data, genfromtxt(audio_file + ".mfcc_d1.csv", delimiter=','), axis=1)
        my_data = np.append(my_data, genfromtxt(audio_file + ".mfcc_d2.csv", delimiter=','), axis=1)

        my_data = np.append(my_data, np.expand_dims(genfromtxt(audio_file + ".energy_d1.csv", delimiter=','), axis=1), axis=1)
        my_data = np.append(my_data, np.expand_dims(genfromtxt(audio_file + ".energy_d2.csv", delimiter=','), axis=1), axis=1)

        # Previous codes creates csv file for features to load numpy array. After that, we can 
        # remove them.
        os.remove(audio_file + ".mfcc.csv")
        os.remove(audio_file + ".mfcc_d1.csv")
        os.remove(audio_file + ".mfcc_d2.csv")
        os.remove(audio_file + ".energy_d1.csv")
        os.remove(audio_file + ".energy_d2.csv")

        np.save("./feature_storage/" + filename, my_data)

        return my_data

    elif (featureplan=="mfcc.txt"):
        cmd = "yaafe -c " + featureplan + " -r 16000 "  + audio_file + " -p Precision=6 -p Metadata=False -n"
        print (cmd)
	
        os.system(cmd)
        filename = (audio_file.split("/")[-1]).split(".")[0]

        my_data = genfromtxt(audio_file + ".mels.csv", delimiter=',')

        os.remove(audio_file + ".mels.csv")
        np.save('./feature_storage/' + filename, my_data)

        return my_data
    else:
        print ("""System is compatible with just mfcc and pyannote_based feature extraction. 
            If you want to create
            new type of feature extraction, please, firstly change this script. For more information
            you can check readme.""")

if __name__ == "__main__":
    create_numpy_for_audio(sys.argv[1], sys.argv[2])
