from glob import glob
from skimage.transform import resize
#import matplotlib.pyplot as plt
#from skimage import io
import re
import time
import os
import cv2

images=[]

dirname = os.path.join('/Users/alfredo-castro/Downloads/', 'animals_10')
imgpath = dirname + os.sep  
start = time.time()

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):  # type: ignore
            filepath = os.path.join(root, filename)
            image = cv2.imread(filepath)
            image_resized = cv2.resize(image, (128, 72))
            cv2.imwrite(filepath.replace('animals_10', 'animals_resized'), image_resized)

print('Total time:', time.time() - start)
    