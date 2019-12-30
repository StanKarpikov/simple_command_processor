import tensorflow as tf
import tensorflow .keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# image_paths, lbls = ["selfies-data/1", "selfies-data/2"], [0., 1.]
#
# labels = []
# file_names = []
# for d, l in zip(image_paths, lbls):
#     # get the list all the images file names
#     name = [os.path.join(d,f) for f in os.listdir(d)]
#     file_names.extend(name)
#     labels.extend([l] * len(name))
#
# file_names = tf.convert_to_tensor(file_names, dtype=tf.string)
# labels = tf.convert_to_tensor(labels)
#
# dataset = tf.data.Dataset.from_tensor_slices((file_names, labels))

input_size_x = 62
input_size_y = 129
directory = 'spectrograms/'
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(dtype=np.float32)
train_generator = img_gen.flow_from_directory(
  directory=directory+'train',
  target_size=(input_size_x, input_size_y),
  color_mode="grayscale",
  batch_size=32,
  class_mode="categorical",
  shuffle=True
)
validate_generator = img_gen.flow_from_directory(
  directory+'validate',
  target_size=(input_size_x, input_size_y),
  color_mode="grayscale",
  batch_size=32,
  class_mode="categorical",
  shuffle=True
)

if False:
    spectrogram = train_generator.next()
    spectrogram = spectrogram[0]
    spectrogram = spectrogram[0]
    spectrogram = spectrogram[:,:,0]
    times = np.linspace(start = 0, stop = 1, num=input_size_y)
    frequencies = np.linspace(start = 0, stop = 8000, num=input_size_x)
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
    model.add(layers.Flatten(input_shape=(input_size_x, input_size_y, 1)))
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dense(units=256, activation='relu'))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=2, activation='softmax'))

    # Configure a model for categorical classification.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01), #keras.optimizers.Adam(lr=1e-6)
                  loss=['categorical_crossentropy'],
                  metrics=['accuracy']) #tf.keras.metrics.CategoricalAccuracy()
    model.summary()
    history = model.fit(train_generator, epochs=5, batch_size=32,
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

#tf.keras.utils.plot_model(model, 'model.png', show_shapes=True)

im = Image.open('spectrograms/test/8056e897_nohash_0.tif')
imarray = np.array(im)
prediction = model.predict(imarray.reshape(1, input_size_x, input_size_y, 1))
print('Prediction:', prediction)
im.show()

#result = model.predict(data, batch_size=32)
#print(result)

