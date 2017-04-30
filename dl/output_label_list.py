import os, csv, time
import glob

listFile = 'image_list.txt'

f = open(listFile, 'w')
for filename in glob.glob("/home/ubuntu/miyamoto/sample/darknet1/data/826_1169_2/images/*.jpg"):
    print(filename)
    f.write(filename+"\n")
f.close()
