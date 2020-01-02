import tensorflow as tf
import tensorflow .keras.layers as layers
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
import os
import custom_data_generator
import cv2
matplotlib.use('TkAgg')

input_size_x = 62
input_size_y = 129
directory = 'spectrograms/'
#img_gen = tf.keras.preprocessing.image.ImageDataGenerator(dtype=np.float32)
img_gen = custom_data_generator.DataGenerator()
train_generator = img_gen.flow_from_directory(
  directory=directory+'train',
  target_size=(input_size_x, input_size_y),
  color_mode="grayscale",
  batch_size=32,
  class_mode="categorical",
  shuffle=True
)
categories = train_generator.get_categories()
validate_generator = img_gen.flow_from_directory(
  directory+'validate',
  target_size=(input_size_x, input_size_y),
  color_mode="grayscale",
  batch_size=32,
  class_mode="categorical",
  shuffle=True
)
test_generator = img_gen.flow_from_directory(
  directory+'test',
  target_size=(input_size_x, input_size_y),
  color_mode="grayscale",
  batch_size=32,
  class_mode="categorical",
  shuffle=True
)
if True:
    for i in range(np.random.randint(1,100)):
        spectrogram = train_generator.next()
    spectrogram = spectrogram[0]
    spectrogram = spectrogram[0]
    #spectrogram = spectrogram[:,:,0]
    times = np.linspace(start = 0, stop = 1, num=input_size_x)
    frequencies = np.linspace(start = 0, stop = 8000, num=input_size_y)
    plt.pcolormesh(times, frequencies, spectrogram)
    #plt.imshow(img) #, aspect=len(times)/len(frequencies)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    exit(0)

if False: # True - Fit new model or False - load from disk
    model = tf.keras.models.Sequential()
    # model.add(layers.Conv2D(input_shape=(input_size_x, input_size_y, 1), filters=32, kernel_size=(3, 3),  activation='relu'))
    # model.add(layers.MaxPool2D(pool_size = (2, 2)))
    # model.add(layers.Conv2D(filters=64, kernel_size=(3, 3),  activation='relu'))
    # model.add(layers.MaxPool2D(pool_size = (2, 2)))
    # model.add(layers.Conv2D(filters=128, kernel_size=(3, 3),  activation='relu'))
    # model.add(layers.MaxPool2D(pool_size = (2, 2)))
    # model.add(layers.Conv2D(filters=256, kernel_size=(3, 3),  activation='relu'))
    # model.add(layers.MaxPool2D(pool_size = (2, 2)))
    # model.add(layers.Dropout(0.25))
    model.add(layers.Flatten(input_shape=(input_size_y, input_size_x)))
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dense(units=256, activation='relu'))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=3, activation='softmax'))

    # Configure a model for categorical classification.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01), #keras.optimizers.Adam(lr=1e-6)
                  loss=['sparse_categorical_crossentropy'],
                  metrics=['accuracy']) #tf.keras.metrics.CategoricalAccuracy()
    model.summary()
    history = model.fit(train_generator, epochs=16, batch_size=32,
              validation_data=validate_generator)
    # Plotting Results

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.title('Training and validation accuracy')
    plt.legend()
    fig = plt.figure()
    fig.savefig('acc.png')

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')

    plt.legend()
    plt.show()

    #model.save_weights('./checkpoints/model_checkpoint')
    # Save:
    # - model weight values
    # - configuration (architecture)
    # - Optimizer configuration
    model.save('./models/model.h5') #HDF5 standart
else:
    model = tf.keras.models.load_model('./models/model.h5')
    #model.load_weights('./checkpoints/model_checkpoint')
    model.summary()

model.evaluate(test_generator, verbose=1)

#tf.keras.utils.plot_model(model, 'model.png', show_shapes=True)
prefix = os.getcwd()
folder_base     = prefix+'/spectrograms'
folder_on       = '/on'
folder_off      = '/off'
folder_other      = '/other'
folder_test     = '/test'
test_folder_on = folder_base + folder_test + folder_on
test_folder_off = folder_base + folder_test + folder_off
test_folder_other = folder_base + folder_test + folder_other

for folder_check in (test_folder_on, test_folder_off, test_folder_other):
    for filename in os.listdir(folder_check):
        file_path = os.path.join(folder_check, filename)
        if os.path.isfile(file_path):
            img = cv2.imread(file_path, -1)
            img = cv2.resize(img, (input_size_x,input_size_y), interpolation=cv2.INTER_AREA)
            print('Test %s'%file_path)
            imarray = np.array(img)
            prediction = model.predict(imarray.reshape(1, input_size_y, input_size_x))
            print('Prediction %s:    %.1f'%( categories[0], prediction[0][0]))
            print('Prediction %s:   %.1f'%( categories[1] , prediction[0][1]))
            print('Prediction %s: %.1f'%( categories[2], prediction[0][2]))

#result = model.predict(data, batch_size=32)
#print(result)

