At this method, we use BiLSTM based DNN. To reproduce the results:

- Firstly, we need to create a folder which stores the numpy array which represent the features of wav files. 
    - These wav files should be 16000 Hz. 
    - We use 25 ms as a window length and 10 ms as a hop length.

- Now, we need to create the ground truth text files. For that, we need to download .mdtm files for AMI corpus. And create a empty folder to stores these .txt files.
    - mkdir txt_files

    After that, we will run these command.

    - python3 ground_truth_txt.py dev.mdtm
    - python3 ground_truth_txt.py .mdtm
    - python3 ground_truth_txt.py .mdtm