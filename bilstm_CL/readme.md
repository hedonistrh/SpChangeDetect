At this method, we use BiLSTM based DNN. To reproduce the results:

- Firstly, we need to create a folder which stores the numpy array which represent the features of wav files. 
    - These wav files should be 16000 Hz. 
    - We use 25 ms as a window length and 10 ms as a hop length.

- Now, we need to create the ground truth text files. For that, we need to download .mdtm files for AMI corpus. 
``` sh
!wget https://raw.githubusercontent.com/pyannote/pyannote-db-odessa-ami/master/AMI/data/speaker_diarization/dev.mdtm

!wget https://raw.githubusercontent.com/pyannote/pyannote-db-odessa-ami/master/AMI/data/speaker_diarization/trn.mdtm

!wget https://raw.githubusercontent.com/pyannote/pyannote-db-odessa-ami/master/AMI/data/speaker_diarization/tst.mdtm
```

And create a empty folder to stores these .txt files.
``` sh
mkdir txt_files
``` 
 After that, we will run these command.
``` sh
python3 ground_truth_txt.py dev.mdtm
python3 ground_truth_txt.py trn.mdtm
python3 ground_truth_txt.py tst.mdtm
```
Now, we can train the system. Example usage:

``` sh
python3 train_model.py "./feature_storage/" "mfcc.txt" 0 4 True 120 True 5
```

Now, we can create the prediction.

``` sh
python3 create_prediction.py EN2001b "mfcc.txt" 0.7
```