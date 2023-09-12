import os
import re
import numpy as np
from glob import glob
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential#,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

dirname = os.path.join('./', 'sportimages')
imgpath = dirname + os.sep 
 
images = []
directories = []
dircount = []
prevRoot=''
cant=0
 
print("reading images... ",imgpath)
 
for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
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

labels=[]
indice=0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice=indice+1
print("Total labels: ",len(labels))
 
sports_=[]
index=0
for file in directories:
    name = file.split(os.sep)
    print(index , name[len(name)-1])
    sports_.append(name[len(name)-1])
    index += 1
 
y = np.array(labels)
X = np.array(images, dtype = np.uint8) #convierto de lista a numpy
 
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
 
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
 
print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

INIT_LR = 1e-2
epochs = 5
batch_size = 64

sports_model = Sequential()
sports_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(21,28,3)))
sports_model.add(LeakyReLU(alpha=0.1))
sports_model.add(MaxPooling2D((2, 2),padding='same'))
sports_model.add(Dropout(0.5))
sports_model.add(Flatten())
sports_model.add(Dense(32, activation='relu'))
sports_model.add(Dense(32, activation='linear'))
sports_model.add(LeakyReLU(alpha=0.1))
sports_model.add(Dropout(0.5)) 
sports_model.add(Dense(nClasses, activation='softmax'))
 
sports_model.summary()
 
sports_model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
sports_train_dropout = sports_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
 
# Saving the network
sports_model.save("./sports_v0.h5py")

test_eval = sports_model.evaluate(test_X, test_Y_one_hot, verbose=1)
 
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])