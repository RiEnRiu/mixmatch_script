# mixmatch_script
Some script to train an semi-labelled dataset with mixmatch and then label unlabelled images.

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
```bash
   "cifar10" is dataset name.Unlabelled-images should be put in "UNLABEL".Others folders are named known-label.Just labelling 10% of the dataset first can you run those scripts and label other unlabelled.

   --- cifar10
   ---|--- airplane
   ---|--- automobile
   ---|--- bird
   ---|--- cat
   ---|--- deer
   ---|--- dog
   ---|--- frog
   ---|--- horse
   ---|--- ship
   ---|--- truck
   ---|--- UNLABEL
```

#### 2. Git clone MixMatch and install dependencies
   
#### 3. python train_and_label_image.py --dir=$DATASET_PATH$
    a] Make your dataset into tfrecord.
	b] Train with tfrecord.
	c] Label images in UNLABEL.
	d] Rerun while UNLABEL is not empty.
