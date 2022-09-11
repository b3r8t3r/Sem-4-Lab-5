import os

# Устранение вывода ошибки

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

import matplotlib.pyplot as plt

from keras.datasets import mnist  # библиотека базы выборок Mnist

from tensorflow import keras

from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

# Загрузка данных

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных

x_train = x_train / 255

x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)

y_test_cat = keras.utils.to_categorical(y_test, 10)

# вывод первых 25  

plt.figure(figsize=(10, 5))

for i in range(25):
    plt.subplot(5, 5, i + 1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(x_train[i], cmap=plt.cm.binary)

plt.show()

x_train = np.expand_dims(x_train, axis=3)

x_test = np.expand_dims(x_test, axis=3)

print(x_train.shape)

# Создание свёрточной модели ИНС

model = keras.Sequential([

    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),

    MaxPooling2D((2, 2), strides=2),

    Conv2D(64, (3, 3), padding='same', activation='relu'),

    MaxPooling2D((2, 2), strides=2),

    Flatten(),

    Dense(64, activation='relu'),

    Dense(64, activation='relu'),

    Dense(10, activation='softmax')

])

print(model.summary())  # вывод структуры НС в консоль

model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])

model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

model.evaluate(x_test, y_test_cat)

n = 1

x = np.expand_dims(x_test[n], axis=0)

res = model.predict(x)

print(res)

print(np.argmax(res))

plt.imshow(x_test[n], cmap=plt.cm.binary)

plt.show()

# Распознавание всей тестовой выборки

pred = model.predict(x_test)

pred = np.argmax(pred, axis=1)

print(pred.shape)

print(pred[:20])

print(y_test[:20])

# Выделение неверных вариантов

mask = pred == y_test

print(mask[:10])

x_false = x_test[~mask]

y_false = x_test[~mask]

print(x_false.shape)

# Вывод первых 25 неверных результатов

plt.figure(figsize=(10, 5))

for i in range(25):
    plt.subplot(5, 5, i + 1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(x_false[i], cmap=plt.cm.binary)

plt.show()