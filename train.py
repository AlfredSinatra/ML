import os
import re
import numpy as np
from glob import glob
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

dirname = os.path.join('/Users/alfredo-castro/Downloads/', 'animals_resized_2')
imgpath = dirname + os.sep 
 
images = []
directories = []
dircount = []
prevRoot=''
cant=0
 
print("reading images... ",imgpath)
 
for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):  # type: ignore
            cant=cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            image = resize(image, (36, 64),anti_aliasing=True,clip=False,preserve_range=True)
            images.append(image)
            b = "Reading..." + str(cant)
            print (b, end="\r")
            if prevRoot !=root:
                print(root, cant)
                prevRoot=root
                directories.append(root)
                dircount.append(cant)
                cant=0
               
dircount.append(cant)
print(dircount) 
dircount = dircount[1:]
dircount[0] = dircount[0]+1

print('Directories:',len(directories))
print("Images in each directory", dircount)
print('Total images in subdirs:',sum(dircount))

labels = []
index = 0
for count in dircount:
    for i in range(count):
        labels.append(index)
    index += 1
print("Total labels: ", len(labels))
 
animals_ = []
index = 0
for file in directories:
    name = file.split(os.sep)
    print(index , name[len(name)-1])
    animals_.append(name[len(name)-1])
    index += 1
 
y = np.array(labels)
X = np.array(images, dtype = np.uint8)  # type: ignore
 
# Find the unique numbers from the train labels
classes = np.unique(y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

#Create de train and testing groups
train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.2)
print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)
 
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.
 
# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)
 
# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])
 
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.3, random_state=23)
 
print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

INIT_LR = 1e-2
epochs = 5
batch_size = 50

animals_model = Sequential()
animals_model.add(Conv2D(32, kernel_size = (2, 2),activation = 'linear', padding = 'same', input_shape=(36, 64, 3)))
animals_model.add(LeakyReLU(alpha = 0.1))
animals_model.add(MaxPooling2D((2, 2), padding = 'same'))
animals_model.add(Dropout(0.5))
animals_model.add(Flatten())
animals_model.add(Dense(1024, activation = 'relu'))
animals_model.add(Dense(512, activation = 'relu'))
animals_model.add(Dense(1024, activation = 'relu'))
animals_model.add(Dense(512, activation = 'relu'))
animals_model.add(Dense(1024, activation = 'relu'))
animals_model.add(Dense(512, activation = 'relu'))
#animals_model.add(Dense(128, activation = 'relu'))
#animals_model.add(Dense(128, activation = 'linear'))
#animals_model.add(Dense(128, activation = 'linear'))
#animals_model.add(Dense(128, activation = 'sigmoid'))
#animals_model.add(Dense(128, activation = 'sigmoid'))
#animals_model.add(Dense(128, activation = 'sigmoid'))
#animals_model.add(Dense(128, activation = 'sigmoid'))
#animals_model.add(Dense(128, activation = 'sigmoid'))
#animals_model.add(Dense(128, activation = 'sigmoid'))
#animals_model.add(Dense(128, activation = 'sigmoid'))
#animals_model.add(Dense(128, activation = 'sigmoid'))
animals_model.add(LeakyReLU(alpha = 0.1))
animals_model.add(Dropout(0.5)) 
animals_model.add(Dense(nClasses, activation = 'softmax'))
 
animals_model.summary()
 
animals_model.compile(loss = 'categorical_crossentropy', optimizer = 'adagrad', metrics = ['accuracy'])
animals_train_dropout = animals_model.fit(train_X, train_label, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (valid_X, valid_label))
 
# Saving the network
animals_model.save("./animals_v0.h5py")

test_eval = animals_model.evaluate(test_X, test_Y_one_hot, verbose=1)
 
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])