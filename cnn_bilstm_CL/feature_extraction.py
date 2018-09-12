from create_numpy_for_audio import create_numpy_for_audio
import sys
import glob
import os

def feature_extraction(root_dir, featureplan):
    """With this function, we will store the each file's mfcc feature extraction
    as a numpy array.

    Arguments:
    root_dir: which folder includes the audio files.
    featureplan: Which txt will be used to express features"""
    root_dir = glob.glob(os.path.join(root_dir, "*wav"))
    for single_file in root_dir:
        create_numpy_for_audio(single_file, featureplan=featureplan)

if __name__ == "__main__":
    feature_extraction(sys.argv[1], sys.argv[2])
