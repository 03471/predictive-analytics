import os
import json
import numpy as np
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical
#
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
import keras.backend as K
import tensorflow as tf
from keras.callbacks import EarlyStopping

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


# server = tf.train.Server.create_local_server()
# sess = tf.Session(server.target)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
K.set_session(sess)

early_stopping_monitor= EarlyStopping(monitor='val_loss', patience=5)
dirpath = os.path.dirname(__file__)
file_path = os.path.join(dirpath, 'shipsnet.json')

try:
    f = open(file_path)
    dataset = json.load(f)
    f.close()
except OSError:
    print('cannot open. check file exists', file_path)
    raise

# Convert data from list to np.array.
data = np.array(dataset['data']).astype('uint8')
labels = np.array(dataset['labels']).astype('uint8')

# View shape of data
print(data.shape)
print(labels.shape)

# Normalize data and transform data for keras model training.
x = data / 255.
x = x.reshape([-1, 3, 80, 80]).transpose([0, 2, 3, 1])
print(x.shape)

y = to_categorical(labels, num_classes=2)
print('labels: ' + str(y.shape))

# View image and labels.
img_id_to_check = np.random.randint(0, x.shape[0] - 1)
im = x[img_id_to_check]

print(img_id_to_check)
print(y[img_id_to_check])

# plt.imshow(im)
# plt.show()

# Create sequential model.
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(80, 80, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# stochastic gradient descent optimizer
# Includes support for momentum, learning rate decay, and Nesterov momentum.
# sgd+Nesterov for shallow networks
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd , metrics=['accuracy'])

# Train model with 20 % of data used for validation.

history = model.fit(x, y, batch_size=80, epochs=20, validation_split=0.2, shuffle= False, callbacks=[early_stopping_monitor], verbose= 1)

# View model training accuracy graph.
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy - hidden layer:2 epochs=20, lr=0.01, batch_size=64, optimizer=sgd')
plt.ylabel('accuracy, ')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# View model training loss graph.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss - hidden layer:2 epochs=20, lr=0.01, batch_size=80, optimizer=sgd')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
