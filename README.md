# mixmatch_script
Mixmatch scripts help train an semi-labelled dataset and then auto-label unlabelled images. You can also get an trained model.

MixMatch code: https://github.com/google-research/mixmatch

Code for the paper: "[MixMatch - A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)" by David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver and Colin Raffel.

## Install dependencies

    sudo git clone https://github.com/google-research/mixmatch.git
    sudo apt install python3-dev python3-virtualenv python3-tk imagemagick
    virtualenv -p python3 --system-site-packages env3
    . env3/bin/activate
    pip install -r mixmatch/requirements.txt
    pip install opencv-python

### Run and label image

#### 1. Label 10% and organize dataset as below
    |-- cifar10
        |-- airplane
        |-- automobile
        |-- bird
        |-- cat
        |-- deer
        |-- dog
        |-- frog
        |-- horse
        |-- ship
        |-- truck
        |-- UNLABEL

"cifar10" is dataset name. Unlabelled-images should be put in "UNLABEL". Others folders are named known-label. Just labelling 10% of the dataset first can you run those scripts to auto-label the rest.

#### 2. Git clone MixMatch and install dependencies

#### 3. Run train_and_label_image.py
    python3 train_and_label_image.py --dir=$DATASET_PATH$
```$DATASET_PATH$``` is path to your dataset.
```train_and_label_image.py``` main process: 
a] Make your dataset into tfrecord.
b] Train with tfrecord.
c] Label images in UNLABEL.
d] You can check right or not and move to labelled dir.
e] Rerun if UNLABEL is not empty.

