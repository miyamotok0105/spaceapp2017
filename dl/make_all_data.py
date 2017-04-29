#!/usr/bin/python
#coding:utf-8
from PIL import Image
# import urllib2 as urllib
import urllib2
import io
import os
import cv2
import time

def get_map_image(long_map, lat_map):
    # url = "http://maps.googleapis.com/maps/api/staticmap?sensor=false&size=256x276&maptype=satellite&center=%s,%s&zoom=15"%(long_map, lat_map)
    url = "http://maps.googleapis.com/maps/api/staticmap?sensor=false&size=256x256&maptype=satellite&center=%s,%s&zoom=15"%(long_map, lat_map)

    request = urllib2.Request(url)
    request.add_header('User-agent', 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)')
    iim = urllib2.urlopen(request)

    # headers = { "User-Agent" :  "Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)" }
    # req = urllib.request.Request(url, None, headers)
    # iim = urllib.request.urlopen(req)
    image_file = io.BytesIO(iim.read()) 
    im = Image.open(image_file).convert('RGBA')
    im.save("image/all_data/%s_%s.png"%(long_map, lat_map), "PNG")
    print(im)
    # box = (0, 256, 256, 256)
    # im2 = im.crop(box)
    # im2.save("test2.png")

#http://maps.googleapis.com/maps/api/staticmap?sensor=false&size=256x276&maptype=satellite&center=40.714728,-73.998672&zoom=15
#http://maps.googleapis.com/maps/api/staticmap?sensor=false&size=640x400&maptype=satellite&visible=-180,-90&visible=-177,-84
#http://maps.googleapis.com/maps/api/staticmap?sensor=false&size=640x400&maptype=satellite&visible=29.64,-13.09&visible=27.38,-18.53
#-180 180
#-90 90
for long_map in range(-177,181, 10):
    for lat_map in range(-90 ,91, 10):
        print long_map
        print lat_map
        if os.path.exists("image/all_data/%s_%s.png"%(long_map, lat_map)) == False:
            get_map_image(long_map, lat_map)
            time.sleep(0.5)
    #     break
    # break











