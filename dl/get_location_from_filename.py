import re
import os
import glob


for name in glob.glob("/home/ubuntu/miyamoto/sample/spaceapp2017/dl/image/all_data/*.png"):

#((-[0-9][0-9])|([0-9][0-9]))_((-[0-9][0-9])|([0-9][0-9])).png
    
    match = re.search(r'((-[0-9]*)|([0-9]*))_((-[0-9]*)|([0-9]*)).png',name)
    file_name = match.group()
    
    #print(name)
    #match = re.findall(r'((-[0-9][0-9])|([0-9][0-9]))',file_name)
    #match = re.search(r'((-[0-9][0-9])|([0-9][0-9]))',file_name)
    print(file_name)
    
    match = re.search(r'^((-[0-9]*)|([0-9]*))',file_name)
    print(match.group())

    match = re.search(r'_((-[0-9]*)|([0-9]*))',file_name)
    print(match.group().replace("_", ""))

    #print(match.group(1))
    #print(match.group(2))
