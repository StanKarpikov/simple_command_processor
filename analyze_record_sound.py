import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import numpy as np
import sounddevice as sd
import queue

import matplotlib.pyplot  as plt
import matplotlib

def plot_pause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

plt.ion()
plt.subplots()
plt.show(block=False)

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
            #sd.play(samples,sample_rate)
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
            arg = np.argmax(prediction)
            probability = prediction[0][arg]
            if probability > 0.9:
                if arg == 0:
                    print('on:   %.2f'%(probability))
                elif arg == 2:
                    print('off:  %.2f' % (probability))
                else:
                    print('+')
            else:
                print('-')

            if True:
                times = np.linspace(start=0, stop=1, num=input_size_x)
                frequencies = np.linspace(start=0, stop=8000, num=input_size_y)
                plt.pcolormesh(times, frequencies, spectrogram)
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plot_pause(0.1)

except KeyboardInterrupt:
    pass
except Exception as e:
    exit(type(e).__name__ + ': ' + str(e))