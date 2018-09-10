At this method, we use BiLSTM based DNN. To reproduce the results (at Linux):

- Firstly, we install Anaconda. Currently (13.08.2018), final version is 5.2.0
``` sh
wget -c https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh
chmod +x Anaconda3-5.2.0-Linux-x86_64.sh
bash ./Anaconda3-5.2.0-Linux-x86_64.sh -b -f -p /usr/local
```

- To use same conda environment.
``` sh
    conda create --name bilstm --file spec-file.txt
    source activate bilstm
```

- We need to create a folder which stores the numpy array which represent the features of wav files. 

    - These wav files should be 16000 Hz. 
    - We use 25 ms as a window length and 10 ms as a hop length.
    ``` sh
    mkdir feature_storage
    ```

    - If you do not have AMI Corpus, you need to download.
    ``` sh
    wget http://groups.inf.ed.ac.uk/ami/download/temp/amiBuild-125026-Mon-Sep-10-2018.wget.sh
    chmod +x amiBuild-125026-Mon-Sep-10-2018.wget.sh
    ./amiBuild-125026-Mon-Sep-10-2018.wget.sh
    ```

    - Now, we can extract features.
    ``` sh
        python3 feature_extraction root_dir featureplan

        python3 feature_extraction "./ami_corpus/*/audio/" "mfcc.txt" {example usage}
    ```
        
- Now, we need to create the ground truth text files. For that, we need to download .mdtm files for AMI corpus. 
``` sh
wget https://raw.githubusercontent.com/pyannote/pyannote-db-odessa-ami/master/AMI/data/speaker_diarization/dev.mdtm

wget https://raw.githubusercontent.com/pyannote/pyannote-db-odessa-ami/master/AMI/data/speaker_diarization/trn.mdtm

wget https://raw.githubusercontent.com/pyannote/pyannote-db-odessa-ami/master/AMI/data/speaker_diarization/tst.mdtm
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
Now, we can train the system. 

``` sh
python3 train_model.py root_dir featureplan how_many_repeat how_many_step boost how_many_boost fuzzy epoch 

python3 train_model.py "./feature_storage/" "mfcc.txt" 10 30 True 120 True 2 {example usage}
```

Now, we can create the prediction.

``` sh
python3 create_prediction.py EN2001b "mfcc.txt" 0.7
```

Lastly, to convert from txt to internal metadata format.

``` sh
python2 metadata_converter.py input_directory output_directory --outputType=mpeg7 --inputType=txt_file

python2 metadata_converter.py testdata/ami_pred out/ami_mpeg7_pred --outputType=mpeg7 --inputType=txt_file {example usage}

```
