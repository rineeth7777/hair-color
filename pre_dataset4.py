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
def resize_and_save(filename,output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    global  COUNT
    global COUNT2
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method

    s=filename.split('\\')[-1]
    #print('sss',s)
    s2=int(s.split('.')[0].split('_')[0])
    #print(s,s2)

    image = Image.open('D:\\AI\\vision\\data\\masks\\0.1_image\\'+str(s2)+'.jpg')
    imarr = np.asarray(image)

    #if(s2>25638):
    #    print('filename '+s2,filename)
    image2 = Image.open(filename)
    imarr2 = np.asarray(image2)
    imarr2 = imarr2 > 254
    imarr2 = imarr2.astype('int32')

    avgs = np.average(imarr, weights=imarr2, axis=(0, 1))
    img_w, img_h = SIZE, SIZE
    data1 = np.full((img_h, img_w), avgs[0])
    data2 = np.full((img_h, img_w), avgs[1])
    data3 = np.full((img_h, img_w), avgs[2])
    data = np.stack((data1, data2, data3), axis=2)
    data = data.astype('uint8')
    img = Image.fromarray(data, mode='RGB')
    img.save(output_dir+'\\'+str(s2)+'.jpg')



if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)
    fol='14'

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'once')
    img_data=os.path.join('D:\\AI\\vision\\data\\masks\\img')

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith('.png')]

    '''check = os.listdir('D:\\AI\\vision\\data\\masks\\0.1_color')
    check = [os.path.join(train_data_dir, f) for f in check if f.endswith('.jpg')]
    check=[int(f.split('\\')[-1].split('.')[0])  for f in check]
    #print(check)
    fn=[int(f.split('\\')[-1].split('.')[0].split('_')[0])  for f in filenames]
    print('--------------------------------------------------------------------------------------------------------------------')
    #print(fn)
    kk=list(set(fn)-set(check))
    kk.sort()
    print(len(kk),kk)'''

    # Split the images in 'train_signs' into 80% train and 20% dev
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    filenames.sort()

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
    output_dir_split = os.path.join(args.output_dir,'{}_color'.format(0.1) )#'{}_annotation'.format(PERCENT)
    if not os.path.exists(output_dir_split):
        os.mkdir(output_dir_split)
    else:
        print("Warning: dir {} already exists".format(output_dir_split))

    #print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
    for filename in tqdm(filenames):
        resize_and_save(filename, output_dir_split, size=SIZE)
    print("Done building dataset")
    #27864 for 0.1