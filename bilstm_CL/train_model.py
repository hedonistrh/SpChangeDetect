from load_training_data import load_training_data
from create_model import create_model 

def train_model(root_dir, featureplan,
                from_file, to_file,
                boost, how_many, fuzzy,
                epoch):

    """To train the DL model. 
    
    Arguments:
    root_dir: Where numpy array is stored.
    featureplan: Which featureplan will be used
    from_file: Index for first file.
    to_file: Index for last file.
    boost: Boolean value. If it is True, we will consider neighboor frames of 
        change frame as a change frame.
    how_many= If boost is True, how many neighboor frames should be considered
        as a change frame.
    fuzzy: If it is true, we will use fuzzy labelling to reflect neighboor frames.
    epoch: How many epoch.
    """

    model = create_model()

    input_array, output_array = load_training_data(root_dir=root_dir,
                                from_file=from_file,
                                to_file=to_file,
                                featureplan=featureplan,
                                boost=boost, how_many=how_many,
                                fuzzy=fuzzy)
    
    max_len = 800 # how many frame will be taken
    step = 800 # step size.

    input_array_specified = []
    output_array_specified = []

    for i in range (0, input_array.shape[0]-max_len, step):
        single_input_specified = (input_array[i:i+max_len,:])
        single_output_specified = (output_array[i:i+max_len,:])

        input_array_specified.append(single_input_specified)
        output_array_specified.append(single_output_specified)

    output_array_specified = np.asarray(output_array_specified)
    input_array_specified = np.asarray(input_array_specified)


    input_array_specified = np.expand_dims(input_array_specified, axis=4)
    model.fit(input_array_specified, output_array_specified,
        epochs=epoch,
        batch_size=16,
        shuffle=False)

    model.save_weights('bilstm_weights_2DCNN.h5')    

    input_array = []
    output_array = []
    


if __name__ == "__main__":
    train_model(sys.argv[1], sys.argv[2],
                sys.argv[3], sys.argv[4],
                sys.argv[5], sys.argv[6],
                sys.argv[7], sys.argv[8])

