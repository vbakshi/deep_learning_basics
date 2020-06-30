import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_set = train_datagen.flow_from_directory(
        './data/cnn/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
        './data/cnn/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
    

#Building the CNN

cnn = Sequential()

cnn.add(Conv2D(input_shape=(64,64,3), filters = 32 , kernel_size=3,activation='relu' ))
cnn.add(MaxPool2D(pool_size=2, strides=2))

cnn.add(Conv2D(filters = 32 , kernel_size=3,activation='relu' ))
cnn.add(MaxPool2D(pool_size=2, strides=2))

cnn.add(Flatten())

cnn.add(Dense(units=128, activation='relu'))
cnn.add(Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(
    x=train_set,
    validation_data=test_set,
    epochs=25
)

cnn.save("./output/cnn_model")







