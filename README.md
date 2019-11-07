# mixmatch_script
Some script to train an semi-labeled dataset with mixmatch and then label them.

## Install dependencies

```bash
sudo git clone https://github.com/google-research/mixmatch.git
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick
virtualenv -p python3 --system-site-packages env3
. env3/bin/activate
pip install -r mixmatch/requirements.txt
```

### Run and label image

#### 1. Copy your images in ROOT_PATH with directory tree below. 
   "cifar10" is dataset name.
   Unlabelled-images should be put in "UNLABEL".
   Others folders are named known-label. 
   Just labelling 10% of the dataset first can you run those scripts and label other unlabelled.
   ©À©¤©¤ cifar10
   ©¦?? ©À©¤©¤ airplane
   ©¦?? ©À©¤©¤ automobile
   ©¦?? ©À©¤©¤ bird
   ©¦?? ©À©¤©¤ cat
   ©¦?? ©À©¤©¤ deer
   ©¦?? ©À©¤©¤ dog
   ©¦?? ©À©¤©¤ frog
   ©¦?? ©À©¤©¤ horse
   ©¦?? ©À©¤©¤ ship
   ©¦?? ©À©¤©¤ truck
   ©¦?? ©¸©¤©¤ UNLABEL

#### 2. Git clone MixMatch and install dependencies
   
#### 3. python train_and_label_image.py --dir=$DATASET_PATH$
    a] Make your dataset into tfrecord.
	b] Train with tfrecord.
	c] Label images in UNLABEL.
	d] Rerun while UNLABEL is not empty.
