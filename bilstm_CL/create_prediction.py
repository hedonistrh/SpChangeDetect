import more_itertools as mit
import numpy as np
from keras.models import load_model
from create_numpy_for_audio import create_numpy_for_audio

def create_prediction(filename, 
                    threshold, 
                    featureplan):
    
    """"It takes audio file and create prediction via lstm system. If output exceeds
    threshold, we will say there is speaker change.
    
    Arguments:
    filename= Which file will be considered.
    threshold: If prediction exceed this value, we will say there is speaker change
    featureplan: Which txt will be used for yaafe feature extraction.
    
    Outputs:
    prediction_array: It stores prediction value for each frame
    prediction_array_rav: Ravel version of prediction array. We will use it.
    prediction_array_ms = It stores which milisecond we have speaker change point.
    """

    win_len=25
    hop=10
    prediction_vector = []

    model = create_model()
    model.load_weights("bilstm_weights_2DCNN.h5")

    feature_vector = np.load("./feature_storage/" + filename + ".npy")
    
    ix_frame = 0
    
    
    while (ix_frame+799<matrix_of_single_audio.shape[0]):        
    
        prediction = model.predict(np.expand_dims(prediction_vector[ix_frame:ix_frame+800], axis=0))
        
        prediction_vector.append(prediction)
        ix_frame += 800
    
    prediction_vector = np.asarray(prediction_vector)
    print (prediction_vector.shape)

    prediction_array = np.ravel(prediction_vector)


    prediction_array_sec = []
    prediction_array_msec = []
    ix_frame_pred = 0

    for pred in prediction_array:
        if (pred > threshold):
            ms_version = float(win_len + (ix_frame_pred * hop)) # milisecond version to represent end point of first embed            
            prediction_array_msec.append(int(ms_version))
            prediction_array_sec.append(ms_version/1000)

        ix_frame_pred += 1

    prediction_array_tenth_ms = np.asarray(prediction_array_msec)/10

    list_cons = [list(group) for group in mit.consecutive_groups(prediction_array_tenth_ms)]

    prediction_array_msec_smooth = []

    for single_list_cons in list_cons:
        prediction_array_msec_smooth.append(np.mean(single_list_cons)*0.01)

    prediction_array_msec_smooth = np.asarray(prediction_array_msec_smooth)

    which_turn = 0

    for single_mean_s in prediction_array_msec_smooth:
        which_turn += 1

        try:
            start_time = float(prediction_array_msec_smooth[which_turn-1])
            end_time = float(prediction_array_msec_smooth[which_turn])

            if ((start_time+0.5) > end_time):
                mean_s[which_turn] = ((start_time+end_time) / 2)
                mean_s = np.delete(mean_s, which_turn-1)
                which_turn -= 1
                
        except:
            pass

np.savetxt(fname="./prediction_txt/" + filename + "_prediction.txt", 
            X=prediction_array_msec_smooth, 
            delimiter=' ', fmt='%1.3f')

return (prediction_array, prediction_array_msec_smooth)

if __name__ == "__main__":
    create_prediction(sys.argv[1], sys.argv[2],
                sys.argv[3], sys.argv[4])