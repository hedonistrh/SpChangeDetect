import numpy as np


def txt_file_to_matrix (filename, type_of_text, sr, featureplan, hop=10, win_len=25, feature_extractor="yaafe"):
    
    
    """It takes the reference(ground truth) text file or prediction text file (they are in second version) and 
    return output array which represent the which frames has a speaker change point.
    
    Arguments:
    filename: Which file will be considered.
    type_of_text: Is it prediction or reference file.
    sr: Sample rate.
    featureplan: Which txt will be used for yaafe feature extraction.
    hop: Hop length. (for Librosa.)
    win_len: Window length. (for Librosa)
    feature_extractor: Which feature extractor will be used. (Now, we have 2 option as
        Pyannote or Yaafe.)"""
    
    
    try:
        feature_vector = np.load("./feature_storage/" + filename + ".npy")
    except: 
        feature_vector = create_numpy_for_audio(audio_file="rad_bremen_media/" + filename + ".wav", 
                                                        feature_extractor="yaafe", hop=hop, win_len=win_len, 
                                                        featureplan=featureplan, sr=sr)
    
    
    if (type_of_text == "reference"):
        path_for_txt = "./txt_ami_full/" + filename.split(".")[0] + "_full_time.txt" 
        print (path_for_txt)
    
    if (type_of_text == "prediction"):
        path_for_txt = "./prediction_txt/" + filename + "_prediction.txt"
        
    change_seconds = []


    with open(path_for_txt) as f:
        content = f.readlines()
        
    content = [x.strip() for x in content] 
    
    for single_change in content:
        change_seconds.append(single_change)
    
    output_array = np.zeros(feature_vector.shape[0])

    for single_change in change_seconds:
        
        single_change_ms = float(single_change)*1000
        which_start_hop = (single_change_ms-win_len)/hop # now we know, milisecond version of change
                                    # which is located after which_hop paramater
                                    # add 2 and round to up
        start_location = math.ceil(which_start_hop + 1)

        output_array[start_location] = 1.0

    return (output_array)