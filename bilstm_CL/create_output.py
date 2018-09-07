import numpy as np  
import math

def create_output(filename, shape, boost,
                how_many,
                fuzzy):
    """With this function, we will create output for
    a file."""
    
    """Arguments:
    filename: Which txt file include the information
        for speech and non-speech part. This text file
        should include each speaker changes at each line.
    shape: This info is based on mfcc input's shape.
    boost: Boolean value. If it is True, we will consider neighboor frames of 
        change frame as a change frame.
    how_many= If boost is True, how many neighboor frames should be considered
        as a change frame.
    fuzzy: If it is true, we will use fuzzy labelling to reflect neighboor frames.
    """

    # This values are default with my project. If you want to change it
    # you can follow my github repo to understand 
    win_len = 25
    hop = 10

    output_array = np.zeros(shape)

    try:
        with open(filename) as f:
            content = f.readlines()
    except:
        print (filename + " can not be found.")
        raise
    
    changes = [x.strip() for x in content]

    for change in changes:
        
        change_frame = (int(change) - win_len) / hop
        
        ## Also you can use this frame as a center.
        # change_frame = int(change) / hop
        
        change_frame = math.ceil(change_frame)
        
        if (change_frame<0): # to get rid confusion at first frame
            change_frame=0
            
        if (boost):
            if (fuzzy):
                output_array[int(change_frame)] = 1.0
                
                for ix_label in range(1, how_many):
                    output_array[int(change_frame-ix_label)] = 1.0 - (float(ix_label)/how_many)
                    try:
                        output_array[int(change_frame+ix_label)] = 1.0 - (float(ix_label)/how_many)
                    except:
                        pass

            else:
                output_array[int(change_frame-(how_many/2)):int(change_frame+int(how_many/2))] = 1.0

    return output_array