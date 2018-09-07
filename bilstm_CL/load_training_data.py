import numpy as np
from create_output import create_output

def load_training_data(root_dir,
                    from_file, to_file, 
                    featureplan,
                    boost,
                    how_many,
                    fuzzy):

    """With this function, train datas will be transformed to 
    suitable format for DL architecture.

    Arguments:
    root_dir: Where numpy array is stored.
    from_file: Index for first file.
    to_file: Index for last file.
    boost: Boolean value. If it is True, we will consider neighboor frames of 
        change frame as a change frame.
    how_many= If boost is True, how many neighboor frames should be considered
        as a change frame.
    fuzzy: If it is true, we will use fuzzy labelling to reflect neighboor frames.
    """
    
    input_array = []
    output = []
    
    for ix in range(from_file, to_file):
        feature_vector = np.load(root_dir[ix]) 
        feature_array = np.ravel(feature_vector)
        input_array.extend(feature_array)

        filename_txt = (str(root_dir[ix]).split("/")[-1])[:-3] + "txt"
        output_array = create_output("./outputs_txt/" + filename_txt, 
                                          shape=feature_vector.shape[0],
                                          boost=boost,
                                          how_many=how_many,
                                          fuzzy=fuzzy)
                
        output.extend(output_array)
    
    if (featureplan=="mfcc.txt"):
        input_array = np.reshape(input_array, (-1, 40))

    elif (featureplan=="pyannote_based.txt"):
        input_array = np.reshape(input_array, (-1, 59))
        
    else:
        print ("""System is compatible with just mfcc and pyannote_based. If you want to create
            new type of feature extraction, please, firstly change this script. For more information
            you can check readme.""")

        raise Exception ("Unknown Feature Extraction Type.")

    output_array = np.asarray(output)
    output_array = np.expand_dims(output_array, axis=1)

    print ("Mean of output:", np.mean(output_array))

    return (input_array, output_array)