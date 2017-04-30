#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.
1;95;0c
Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images, scale them to 256x256 and convert them to RGB, and make
two lists of space-separated CSV whose first column is full path to image and
second column is zero-origin label (this format is same as that used by Caffe's
ImageDataLayer).

"""
from __future__ import print_function
import argparse
import random

from PIL import Image
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
from chainer import cuda
import chainer.links as L
import chainer.functions as F
import os
import re
import shutil


import alex
import googlenet
import googlenetbn
import nin


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main():
    archs = {
        'alex': alex.Alex,
        'alex_fp16': alex.AlexFp16,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'googlenetbn_fp16': googlenetbn.GoogLeNetBNFp16,
        'nin': nin.NIN
    }

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('--test', help='Path to training image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='nin',
                        help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    # Initialize the model to train
    model = archs[args.arch]()
    if args.initmodel:
        print('Load model from', args.initmodel)
        #chainer.serializers.load_npz(args.initmodel, model)
        chainer.serializers.load_hdf5(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make the GPU current
        model.to_gpu()
    xp = cuda.cupy if args.gpu >= 0 else np

    mean = np.load(args.mean)
    test = PreprocessedDataset(args.test, args.root, mean, model.insize, False)
    print("test----")
#    for line in open('test.txt', 'r'):
#        print(line)

    for t, line in zip(test, open(args.test, 'r')):
        #print(t[0].shape)
        #x = np.asarray(1, t[0])[:].astype(np.float32)
        x = np.ndarray((1, 4, model.insize, model.insize), dtype=np.float32)
        x[0] = t[0]
        x = xp.asarray(x)
        x = chainer.Variable(x, volatile=True) 
                                                                                                                                                              
        y = model.predictor(x)
        score = F.softmax(y)
        #print(line)
        #print(np.argmax(y.data[0].tolist()))
        #print("---")

        if np.argmax(y.data[0].tolist()) == 0:
            #print(line)

            match = re.search(r'((-[0-9]*)|([0-9]*))_((-[0-9]*)|([0-9]*)).png',line)
            file_name = match.group()  
            #file only
            print(file_name)

            shutil.copyfile(os.path.join("/home/ubuntu/miyamoto/sample/spaceapp2017/dl/image/all_data", file_name), os.path.join("/home/ubuntu/miyamoto/sample/spaceapp2017/dl/image/good_place", file_name))


            match1 = re.search(r'^((-[0-9]*)|([0-9]*))',file_name)
            #print(match.group())
            
            match2 = re.search(r'_((-[0-9]*)|([0-9]*))',file_name)
            #print(match.group().replace("_", ""))

            #here is location
            #print(match1.group(), ",", match2.group().replace("_", ""))

if __name__ == '__main__':
    main()

