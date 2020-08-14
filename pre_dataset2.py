"""Split the SIGNS dataset into train/dev/test and resize images to 64x64.

The SIGNS dataset comes in the following format:
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...

Original images have size (3024, 3024).
Resizing to (64, 64) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and dev sets.
Because we don't have a lot of images and we want that the statistics on the dev set be as
representative as possible, we'll take 20% of "train_signs" as dev set.
"""

import argparse
import random
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


SIZE = 256
PERCENT=0.1
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='D:\\AI\\vision\\data\\masks', help="Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default='D:\\AI\\vision\\data\\masks', help="Where to write the new data")

COUNT=0
COUNT2=0
def resize_and_save(filename, imgnames,output_dir,out_img, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    global  COUNT
    global COUNT2
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method

    s=filename.split('\\')[-1]
    #print('sss',s)
    s2=int(s.split('.')[0].split('_')[0])
    #print(s,s2)
    data=np.asarray(image)
    #val=1-np.count_nonzero(data==0)/(256*256*3)
    val2= np.count_nonzero(data == 255) / (256 * 256 * 3)
    if('D:\\AI\\vision\\data\\masks\\0.1_image\\'+str(s2)+'.jpg' not in imgnames):
        COUNT2=-1
        print('s2mann',s2)
    #print('data shape',1-np.count_nonzero(data==0)/(256*256*3))
    '''if(val2>PERCENT):
        COUNT+=1
        image.save(os.path.join(output_dir, s))
        img=Image.open('D:\\AI\\vision\\data\\masks\\img\\'+str(s2)+'.jpg')
        img = img.resize((size, size), Image.BILINEAR)
        img.save(os.path.join(out_img, str(s2)+'.jpg'))'''

    #image.save(os.path.join('D:\\AI\\vision\\data\\684', filename.split('/')[-1]))



if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)
    fol='14'

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, '0.1_annotation')
    img_data=os.path.join('D:\\AI\\vision\\data\\masks\\img')

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith('.png')]
    imgnames = os.listdir('D:\\AI\\vision\\data\\masks\\0.1_image')
    imgnames = [os.path.join('D:\\AI\\vision\\data\\masks\\0.1_image', f) for f in imgnames if f.endswith('.jpg')]

    # Split the images in 'train_signs' into 80% train and 20% dev
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    imgnames.sort()

    #random.shuffle(filenames)

    #split = int(0.8 * len(filenames))
    #train_filenames = filenames[:split]
    #dev_filenames = filenames[split:]
    #filenames = {'train': train_filenames,
    #             'dev': dev_filenames,
    #             'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test

    #for split in ['train', 'dev', 'test']:
    output_dir_split = os.path.join(args.output_dir,'{}_annotation'.format(PERCENT) )#'{}_annotation'.format(PERCENT)
    if not os.path.exists(output_dir_split):
        os.mkdir(output_dir_split)
    else:
        print("Warning: dir {} already exists".format(output_dir_split))

    output_dirit = os.path.join(args.output_dir,'{}_image'.format(PERCENT) )#'{}_image'.format(PERCENT)
    if not os.path.exists(output_dirit):
        os.mkdir(output_dirit)
    else:
        print("Warning: dir {} already exists".format(output_dirit))

    #print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
    for filename in tqdm(filenames):
        resize_and_save(filename,imgnames, output_dir_split,output_dirit, size=SIZE)
    print('FINAL COUNT', COUNT)
    print('FINAL COUNT STRICT',COUNT2)

    print("Done building dataset")
    #27864 for 0.1