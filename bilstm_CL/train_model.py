from load_training_data import load_training_data
from create_model import create_model 
import sys
import glob
import os
import numpy as np

def train_model(root_dir, featureplan,
                how_many_repeat, how_many_step,
                boost, how_many_boost, fuzzy,
                epoch):

    """To train the DL model. 
    
    Arguments:
    root_dir: Where numpy array is stored.
    featureplan: Which featureplan will be used
    how_many_repeat: Epoch for all files.
    how_many_step: We need to divide to all files into subfolder package. This 
        represent that one package include how many file 
    boost: Boolean value. If it is True, we will consider neighboor frames of 
        change frame as a change frame.
    how_many= If boost is True, how many neighboor frames should be considered
        as a change frame.
    fuzzy: If it is true, we will use fuzzy labelling to reflect neighboor frames.
    epoch: How many epoch for one subfolder package.
    """

    model = create_model(featureplan)

    how_many_file = len(glob.glob(os.path.join(root_dir, "*npy")))
    print ("Total File :", how_many_file)

    # If we want to load all AMI corpus directly to the RAM, we will have
    # the memory problem. So that, we divide to parts.
    how_many_step = int(how_many_step)
    how_many_repeat = int(how_many_repeat)

    ix_repeat = 0 # index

    while(ix_repeat < how_many_repeat):
        ix_repeat += 1
        print ("Repeat: ", ix_repeat)

        ix_step = 0

        from_file = 0
        to_file = how_many_file/(how_many_step)
        print (to_file)

        while(ix_step < how_many_step):
            print ("Step: ", ix_step)
            try:
                input_array, output_array = load_training_data(root_dir=root_dir,
                                        from_file=from_file,
                                        to_file=to_file,
                                        featureplan=featureplan,
                                        boost=boost, how_many=how_many_boost,
                                        fuzzy=fuzzy)
            
                max_len = 800 # how many frame will be taken for one block. It have to 
                            # be same with model's input frame's first parameter.

                step = 800    # step size.

                input_array_specified = []
                output_array_specified = []

                for i in range (0, input_array.shape[0]-max_len, step):
                    single_input_specified = (input_array[i:i+max_len,:])
                    single_output_specified = (output_array[i:i+max_len,:])

                    input_array_specified.append(single_input_specified)
                    output_array_specified.append(single_output_specified)

                output_array_specified = np.asarray(output_array_specified)
                input_array_specified = np.asarray(input_array_specified)

                model.fit(input_array_specified, output_array_specified,
                    epochs=int(epoch),
                    batch_size=2,
                    shuffle=False)
                # if you use big batch_size, you will
                # have a problem about memory.
                model.save_weights('bilstm_weights.h5')   
            except FileNotFoundError:
                print ('Probably, corresponding text file is not in the directory.')
                print ('Pass this epoch.')
                pass

            
            
            input_array = []
            output_array = []
    
            from_file += how_many_file/(how_many_step)
            to_file += (from_file + (how_many_file/(how_many_step)))
            

if __name__ == "__main__":
    train_model(sys.argv[1], sys.argv[2],
                sys.argv[3], sys.argv[4],
                sys.argv[5], sys.argv[6],
                sys.argv[7], sys.argv[8])

