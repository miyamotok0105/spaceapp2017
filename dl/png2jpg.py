# -*- coding: utf-8 -*-
import os
import cv2
import glob
import numpy as np
from PIL import Image, ImageTk
#ubuntu:
#sudo apt-get install python-imaging-tk

for name in glob.glob("/home/ubuntu/miyamoto/sample/chainer/examples/imagenet/300image_jpg/random_place_jpg/*.png"):
    print(name)

    img = Image.open(name)
    if img.mode != "RGB":
        img = img.convert("RGB")
    os.remove(name)
    img.save(name.replace(".png", ".jpg"))





