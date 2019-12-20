#-*-coding:utf-8-*-
import argparse
import os
import sys

from make_tfrecord import scan_images

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dir', type = str,required=True, help = 'Data set path.')
    args = parser.parse_args()

    if not os.path.isdir('./mixmatch'):
        if os.system('git clone https://github.com/google-research/mixmatch.git')!=0:
            sys.exit()
        else:
            sys.exit('Install dependencies mentioned in ReadMe.md')

    train_kimg = None
    labels, len_labeled, len_unlabel = None, None, None
    dataset = os.path.basename(os.path.abspath(args.dir))
    # if 1:
    while 1:
        label_dict, label_cid_list, _, unlabel_cid_list, _ = scan_images(args.dir)
        if label_dict!=labels or len_labeled!=len(label_cid_list) or len_unlabel!=len(unlabel_cid_list):
            labels, len_labeled, len_unlabel = label_dict, len(label_cid_list), len(unlabel_cid_list)
            if len_unlabel == 0:
                sys.exit()
                # break
            os.system('CUDA_VISIBLE_DEVICES=0 python make_tfrecord.py --dir={0}'.format(args.dir))
        if train_kimg is None:
            train_kimg = (len_labeled+len_unlabel)*50
        else:
            train_kimg += (len_labeled+len_unlabel)*5
        # print(train_kimg)
        os.system('ML_DATA=tfrecord CUDA_VISIBLE_DEVICES=0 python mixmatch.py --dataset={0} --w_match=75 --beta=0.75 --filters=32 --lr=0.0002 --train_kimg={1} --report_kimg={2}'.format(dataset, train_kimg, (len_labeled+len_unlabel)*5))
        os.system('ML_DATA=tfrecord CUDA_VISIBLE_DEVICES=0 python move_file.py --dir={0} --high_thresh={1} --low_thresh={2}'.format(args.dir,0.7,0.3))
        


              




