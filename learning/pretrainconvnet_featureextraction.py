import os

from keras import models
from keras import layers
import numpy as np
from keras.applications import VGG16, InceptionResNetV2
from keras.applications.inception_resnet_v2 import decode_predictions, preprocess_input

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.applications import imagenet_utils
from keras.utils import multi_gpu_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import keras
tbcallback = keras.callbacks.TensorBoard(log_dir=os.path.join(os.path.dirname(__file__),'../graph'), histogram_freq=0,
          write_graph=True, write_images=True)


base_dir = 'D:\Documents\PycharmProjects\AsYouWishDL\data'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'validation')

"""
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

conv_base = InceptionResNetV2(include_top=False,
                  input_shape=(150, 150, 3))
#extending the conv_base model and running it end-to-end on the inputs
model = models.Sequential()
model.add(conv_base)

for layer in model.layers[:len(conv_base.layers)]:
    layer.trainable = False
conv_base.trainable = False


model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(layers.Dense(2, activation='softmax'))

"""


num_classes = 2

model = models.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(150, 150, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))



print(model.summary())



print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))




print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))



train_datagen = ImageDataGenerator(
      rescale=1./255,
      #rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=False,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print('Training...')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=20,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=sum((len(x[2]) for x in os.walk(validation_dir)))//20,
      verbose=1,
      callbacks=[tbcallback]
        )


model.save('personalcatmodel.h5py')
#model.save('newnetowrkmodelcats.h5py')


m = load_model('./newnetowrkmodelcats.h5py')
img = load_img('../data/validation/wesley/IMG_0498.JPG')
img = img.resize((150,150))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

print(m.predict(x))
print(m.predict_classes(x))
"""
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
"""
def predictcat(imgfile):
    image = load_img(imgfile, target_size=(150, 150))
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)
    #image = imagenet_utils.preprocess_input(image)

    # classify the image
    preds = model.predict(image)
    # r = imagenet_utils.decode_predictions(preds)
    if preds[0][1] == 1:
        return preds, 'Wesley'
    else:
        return preds, 'Buttercup'


model = load_model(r'D:\Documents\PycharmProjects\AsYouWishDL\learning\newnetowrkmodelcats.h5py')