# -*- coding: utf-8 -*-
"""Artificial Intelligence Gender Prediction with Eyes Software - Emirhan BULUT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WVkK8I9hNFhpvfWY3t5eV7kibtcBMhKf
"""

!unzip eyes-rtte.zip

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2)

train_generator = train_datagen.flow_from_directory('/content/datas',target_size=(64, 64),
                                                    batch_size=128,
                                                    class_mode='categorical',
                                                    color_mode="grayscale",
                                                    subset='training')

test_generator = train_datagen.flow_from_directory('/content/datas',target_size=(64, 64),
                                                    batch_size=128,
                                                    class_mode='categorical',
                                                    color_mode="grayscale",
                                                    subset='validation')

from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import Sequential

model = Sequential(name="Emirhan_Eye_Gender_Detection_Deep_Learning")

model.add(keras.Input(shape=(64, 64, 1)))

model.add(layers.Conv2D(filters=32, kernel_size=(2,2), activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(filters=256, kernel_size=(2,2), activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(filters=128, kernel_size=(2,2), activation="relu"))

model.add(layers.Flatten())

num_classes = 2
model.add(layers.Dropout(0.2))
model.add(layers.Dense(16,activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(num_classes, activation="softmax"))
                                                                                                                                                                                                                                                                                                                                                                                                   
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model_history = model.fit(train_generator,
                          epochs=32,
                          shuffle=True,
                          validation_data=test_generator,
                          callbacks=[
    keras.callbacks.ModelCheckpoint("model/save_at_{epoch}.h5")],
                          batch_size=64)

!unzip test_data.zip

test_data_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_data_datagen.flow_from_directory('/content/test_data',target_size=(64, 64),
                                                    batch_size=8,
                                                    class_mode='categorical',color_mode="grayscale")

from tensorflow.keras.models import load_model

#73/73 [==============================] - 6s 79ms/step - loss: 0.2415 - accuracy: 0.8882 - val_loss: 0.2542 - val_accuracy: 0.8984
#Because This is not overfitting. After from 15 epoch, start overfitting :((
model = load_model('/content/model/save_at_15.h5')

prediction_gender_from_eye = model.predict(test_data)

model.evaluate(test_data)

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('/content/emirhan_eye.jpg')
imgg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = img / 255
img = cv2.resize(img,(64,64))
img = np.reshape(img,[1,64,64,1])
predict_emirhan_eye = model.predict(img)
print("""Your Prediction: """)
if predict_emirhan_eye[0][1] > predict_emirhan_eye[0][0] :
  print("Hey Emirhan! Your gender is Male, man!")
elif predict_emirhan_eye[0][0] > predict_emirhan_eye[0][1] :
    print("Hey Emirhan! Your gender is Female, Matmazel :))!")
plt.imshow(imgg)
plt.show()