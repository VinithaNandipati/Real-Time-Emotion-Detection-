import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

num_classes=7
img_rows,img_cols=48,48
#how many images we want to give the model to train at once
batch_size=8

train_data_dir=r'C:\Users\heman\Desktop\real-time emotion detection system\data\train'
validation_data_dir=r'C:\Users\heman\Desktop\real-time emotion detection system\data\test'

train_data_gen=ImageDataGenerator(rescale=1./255,#Normalizes image pixel values from [0, 255] → [0, 1]
                                  rotation_range=30,#Randomly rotates the image within ±30° during training-Helps the model recognize objects even if they’re slightly rotated
                                  shear_range=0.3,#Applies shear transformations (slanting or tilting the image)
                                  zoom_range=0.3,#Randomly zooms in or out of the image by up to 30%-Helps the model handle objects at different distances
                                  width_shift_range=0.4,#Shifts the image horizontally by up to 40% of its total width.
                                  height_shift_range=0.4,#Shifts the image vertically by up to 40% of its total height
                                  horizontal_flip=True,#Randomly flips the image left-to-right
                                  )

validation_data_gen=ImageDataGenerator(rescale=1./255)

#training the model
train_generator=train_data_gen.flow_from_directory(train_data_dir,
                                                   color_mode='grayscale',
                                                   target_size=(img_rows,img_cols),
                                                   batch_size=batch_size,
                                                   class_mode='categorical',#'categorical' - Converts labels into one-hot encoding
                                                   shuffle=True)

validation_generator=validation_data_gen.flow_from_directory(validation_data_dir,
                                                   color_mode='grayscale',
                                                   target_size=(img_rows,img_cols),
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   shuffle=True)

model=Sequential()

#cnn
#block-1
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(48,48,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(48,48,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block-2
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(48,48,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(48,48,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block-3
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(48,48,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(48,48,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block-4
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(48,48,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(48,48,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block-5
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#block-6
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#block-7
model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

checkpoints=ModelCheckpoint('Emotion_little_vgg.h5',
                            monitor='val_loss', 
                            mode='min',
                            save_best_only=True,
                            verbose=1)

earlystop=EarlyStopping(monitor='val_loss',
                        min_delta=0,
                        patience=3,
                        verbose=1,
                        restore_best_weights=True)

reduce_lr=ReduceLROnPlateau(monitor='val_loss',
                            factor=0.2,
                            patience=3,
                            verbose=1,
                            min_delta=0.0001)

callbacks=[earlystop,checkpoints,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0005),
              metrics=['accuracy'])

nb_train_sample=24176
nb_validation_samples=3006
epochs=25

history=model.fit(train_generator,
                            steps_per_epoch=nb_train_sample//batch_size,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=validation_generator,
                            validation_steps=nb_validation_samples//batch_size)