import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import numpy as np
import sounddevice as sd
import queue

input_size_x = 62
input_size_y = 129

global model
model = tf.keras.models.load_model('./models/model.h5')
model.summary()
q = queue.Queue()

print(sd.query_devices())
sample_rate = 16000
block_duration = 1000 #msec
device = 2

try:
    def callback(indata, frames, time, status):
        if status:
            text = '-- ' + str(status) + ' --'
            print(text)
        if any(indata):
            samples = indata[:, 0]
            frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=256, noverlap=0)
            maximum = np.max(spectrogram)
            print('max %f'%(maximum*1e6))
            if(maximum*1e6 > 0.1):
                spectrogram = np.array(spectrogram / maximum)
                q.put(spectrogram)
        else:
            print('no input')


    with sd.InputStream(device=device, channels=1, callback=callback,
                        blocksize=int(sample_rate * block_duration / 1000),
                        samplerate=sample_rate):
        while True:
            spectrogram = q.get()
            if spectrogram is None:
                continue
            prediction = model.predict(spectrogram.reshape(1, input_size_y, input_size_x))
            if prediction.max() > 0.5:
                if prediction[0][0] > prediction[0][1]:
                    print('on:  %.2f'%(prediction[0][0]))
                else:
                    print('off: %.2f'%(prediction[0][1]))
            else:
                print('-')

except KeyboardInterrupt:
    pass
except Exception as e:
    exit(type(e).__name__ + ': ' + str(e))