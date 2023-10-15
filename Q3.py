#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    r"C:\Users\Prashant\Desktop\Deep\Section+40+-+Convolutional+Neural+Networks+(CNN)\Section 40 - Convolutional Neural Networks (CNN)\dataset\training_set",
    target_size=(224, 224),  
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    r"C:\Users\Prashant\Desktop\Deep\Section+40+-+Convolutional+Neural+Networks+(CNN)\Section 40 - Convolutional Neural Networks (CNN)\dataset\test_set",
    target_size=(224, 224),  
    batch_size=32,
    class_mode='binary'
)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(training_set, validation_data=test_set, epochs=5)


import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

image_path = r"C:\Users\Prashant\Desktop\Deep\Section+40+-+Convolutional+Neural+Networks+(CNN)\Section 40 - Convolutional Neural Networks (CNN)\dataset\single_prediction\cat_or_dog_2.jpg"
test_image = Image.open(image_path).resize((224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = model.predict(test_image / 255.0)
if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

