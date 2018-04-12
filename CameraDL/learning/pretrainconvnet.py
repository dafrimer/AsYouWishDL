"""
RESOURCES

 https://github.com/fchollet/deep-learning-with-python-notebooks
supplementary:L
https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/
https://www.pyimagesearch.com/2017/12/18/keras-deep-learning-raspberry-pi/
"""
base_dir = 'D:\Documents\PycharmProjects\AsYouWishDL\data'


from keras.applications import VGG16
from keras.applications.vgg16 import decode_predictions

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

"""

Running the convolutional base over our dataset, recording its output to a Numpy array on disk, then using this
data as input to a standalone densely-connected classifier similar to those you have seen in the first chapters of
this book. This solution is very fast and cheap to run, because it only requires running the convolutional base once
for every input image, and the convolutional base is by far the most expensive part of the pipeline. However,
for the exact same reason, this technique would not allow us to leverage data augmentation at all.

"""


import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator



train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'validation')



datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, sum((len(x[2]) for x in os.walk(train_dir))))
validation_features, validation_labels = extract_features(validation_dir, sum((len(x[2]) for x in os.walk(validation_dir))))
test_features, test_labels = extract_features(test_dir, sum((len(x[2]) for x in os.walk(test_dir))))


## The extracted features are currently of shape (samples, 4, 4, 512). We will feed them to a densely-connected
# classifier, so first we must flatten them to (samples, 8192):
train_features = np.reshape(train_features, ( sum((len(x[2]) for x in os.walk(train_dir))), 4 * 4 * 512))
validation_features = np.reshape(validation_features, ( sum((len(x[2]) for x in os.walk(validation_dir))), 4 * 4 * 512))
test_features = np.reshape(test_features, ( sum((len(x[2]) for x in os.walk(test_dir))), 4 * 4 * 512))



from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

model.save('simplemodel.h5py')

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

print('DONE')

