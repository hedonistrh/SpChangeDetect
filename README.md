This is the repository for Speaker Change Detection algorithms which are based on different Deep Learning architectures.

Firstly, you can start with [my blogpost](https://hedonistrh.github.io/2018-07-09-Literature-Review-for-Speaker-Change-Detection/) to read summary of different algorithms.

We have tried BiLSTM, CNN+BiLSTM and Speaker2Vec. You can find the detailed Jupyter Notebooks for these approaches [at this repository.](https://github.com/hedonistrh/SpChangeDetect)

Also, you can find the scripted version of BiLSTM and CNN+BiLSTM. If you read their readme and follow whole steps, it should work. (If it does not work, you can open the issue or send the e-mail to [me](herdoganturkey@gmail.com). However, for the visualization, you need to use Jupyter Notebooks. Unfortunately, some function names are different in the script and notebooks. Because, scripted version has little bit different architecture.  For the Speaker2Vec, I have not done the scripted version, however, I am planning to provide that. 

Our results are not so good, however, whole pipeline is ready. So that, you can tweak the paramaters, do fine-tuning and change the DL architecture to get better result.