#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing our required packages Packages
import numpy as np 
import pandas as pd 
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


#Loading our datasets, 
train_df=pd.read_csv('datasets/sign_mnist_train.csv')
test_df=pd.read_csv('datasets/sign_mnist_test.csv')


# In[5]:


#Looking at the basic statistics of our dataset.
train_df.describe()


# In[25]:


#taking a look at our data
train_df.head()


# In[9]:


train_label=train_df['label']
train_label.head()
trainset=train_df.drop(['label'],axis=1)
trainset.head()


# In[10]:


X_train = trainset.values
X_train = trainset.values.reshape(-1,28,28,1)
print(X_train.shape)


# In[11]:


test_label=test_df['label']
X_test=test_df.drop(['label'],axis=1)
print(X_test.shape)
X_test.head()


# In[12]:


#Converting our integers to binary form. With the help of LabelBinarizer
from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
y_train=lb.fit_transform(train_label)
y_test=lb.fit_transform(test_label)


# In[13]:


y_train


# In[14]:


X_test=X_test.values.reshape(-1,28,28,1)


# In[15]:


print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


# In[16]:


'''
Augmenting the image dataset to generate new data

ImageDataGenerator package from keras.preprocessing.image allows to add different distortions to image dataset by providing random rotation, zoom in/out , height or width scaling etc to images pixel by pixel.

Here is the package details https://keras.io/preprocessing/image/

The image dataset in also normalised here using the rescale parameter which divides each pixel by 255 such that the pixel values range between 0 to 1.
'''

train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 0,
                                  height_shift_range=0.2,
                                  width_shift_range=0.2,
                                  shear_range=0,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

X_test=X_test/255


# In[17]:


'''
Visualization of the Dataset
Preview of the images in the training dataset
'''
fig,axe=plt.subplots(2,2)
fig.suptitle('Preview of dataset')
axe[0,0].imshow(X_train[0].reshape(28,28),cmap='gray')
axe[0,0].set_title('label: 3  letter: C')
axe[0,1].imshow(X_train[1].reshape(28,28),cmap='gray')
axe[0,1].set_title('label: 6  letter: F')
axe[1,0].imshow(X_train[2].reshape(28,28),cmap='gray')
axe[1,0].set_title('label: 2  letter: B')
axe[1,1].imshow(X_train[4].reshape(28,28),cmap='gray')
axe[1,1].set_title('label: 13  letter: M')


# In[19]:


'''
Building our CNN Model
This model consist of-

1. Three convolution layer which uses  MaxPooling for better feature capture.
2. A dense layer with 512 units
3. The output layer consist of 24 units for 24 different classes



Some information about the Convolution layers
The activation fucntion which we have used is ReLu.

Conv layer 1 -- UNITS - 128 KERNEL SIZE - 5 * 5 STRIDE LENGTH - 1 ACTIVATION - ReLu
Conv layer 2 -- UNITS - 64 KERNEL SIZE - 3 * 3 STRIDE LENGTH - 1 ACTIVATION - ReLu
Conv layer 3 -- UNITS - 32 KERNEL SIZE - 2 * 2 STRIDE LENGTH - 1 ACTIVATION - ReLu

MaxPool layer 1 -- MAX POOL WINDOW - 3 * 3 STRIDE - 2
MaxPool layer 2 -- MAX POOL WINDOW - 2 * 2 STRIDE - 2
MaxPool layer 3 -- MAX POOL WINDOW - 2 * 2 STRIDE - 2
'''

model=Sequential()
model.add(Conv2D(128,kernel_size=(5,5),
                 strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))
model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
          
model.add(Flatten())


# In[20]:


#Dense and output layers
model.add(Dense(units=512,activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=24,activation='softmax'))
model.summary()


# In[21]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[22]:


#Training the model
model.fit(train_datagen.flow(X_train,y_train,batch_size=200),
         epochs = 35,
          validation_data=(X_test,y_test),
          shuffle=1
         )


# In[23]:


#Evaluating the model
(ls,acc)=model.evaluate(x=X_test,y=y_test)


# In[24]:


print('Model Accuracy is {}%'.format(acc*100))


# In[ ]:




