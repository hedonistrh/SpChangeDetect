from create_numpy_for_audio import create_numpy_for_audio

def feature_extraction(root_dir, featureplan):
    """With this function, we will store the each file's mfcc feature extraction
    as a numpy array.

    Arguments:
    root_dir: which folder includes the audio files.
    featureplan: Which txt will be used to express features"""
    
    for file in root_dir:
        create_numpy_for_audio(file, featureplan=featureplan)

if __name__ == "__main__":
    create_numpy_for_audio(sys.argv[1], sys.argv[2])
