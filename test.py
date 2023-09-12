import time
from glob import glob
import os
import matplotlib.pyplot as plt
#import h5py
#import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage.transform import resize

hdf5_file_path = '/Users/alfredo-castro/Documents/Py_projects/sports_v0.h5py'
games_model = keras.models.load_model(hdf5_file_path)

label_ = [
    'Golf',
    'Tenis',
    'Box',
    'Baseball',
    'Basketball',
    'F1'
    ]

images=[]
filenames_path = './sportimages/box/'
filenames = [obj for obj in sorted(glob(os.path.join(filenames_path, '*.jpg')))]
start = time.time()

for filepath in filenames:
    image = plt.imread(filepath, 0)
    image_resized = resize(image, (21, 28),anti_aliasing=True,clip=False,preserve_range=True)
    images.append(image_resized)

X = np.array(images, dtype=np.uint8) # type: ignore
test_X = X.astype('float32')
test_X = test_X / 255.

predicted_classes = games_model.predict(test_X)
cc = 0
for i, img_tagged in enumerate(predicted_classes):
    acc = max(predicted_classes[:][i])
    if acc > 0.95:
        cc += 1
print('__acc__',cc, len(filenames), 'percentage:', 100*cc/len(filenames))

