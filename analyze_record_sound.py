import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import numpy as np
import sounddevice as sd
import queue
import os
import matplotlib.pyplot as plt
import matplotlib
import time
import speechpy


def process_sound_predict(spectrogram):
    global model
    global input_size_y
    global input_size_x
    prediction = model.predict(spectrogram.reshape(1, input_size_y, input_size_x))
    print(prediction)
    arg = np.argmax(prediction)
    probability = prediction[0][arg]
    # if probability > 0.9:
    if arg == 0:
        print('on:   %.2f' % (probability))
    elif arg == 2:
        print('off:  %.2f' % (probability))
    else:
        print('other')
    # else:
    #   print('?')

def process_sound_save(insamples):
    global index
    timestr = time.strftime("%Y%m%d_%H%M%S")
    dir = directory + '/' + sound_name + '/'
    filename = timestr + '_' + str(index) + '.wav'
    full_filename = dir + filename
    os.makedirs(dir, exist_ok=True)
    wavfile.write(full_filename, sample_rate, insamples)
    print('%s written' % (filename))
    index += 1

def get_sound_spectrogram(insamples):
    if False:
        frequencies, times, spectrogram = signal.spectrogram(insamples, sample_rate, nperseg=256, noverlap=0)
        maximum = np.max(spectrogram)
        spectrogram = np.array(spectrogram / maximum)
        return spectrogram
    else:
        mfcc = speechpy.feature.mfcc(insamples, sampling_frequency=sample_rate, frame_length=0.01,
                                     frame_stride=0.01,
                                     num_filters=40, fft_length=256, num_cepstral=13, low_frequency=0,
                                     high_frequency=None)
        mfcc_cmvn = speechpy.processing.cmvn(mfcc, variance_normalization=True)
        return mfcc_cmvn


def process_sound_plot(spgram):
    global c
    if c is not None:
        c.remove()
    c = plt.pcolormesh(spgram)
    plot_pause(0.1)

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
#input_size_x = 62samples
#input_size_y = 129
input_size_x = 13
input_size_y = 99

model = tf.keras.models.load_model('./models/model.h5')
model.summary()
q = queue.Queue()

prefix = os.getcwd()
directory = prefix + '/my_speech_database'
sound_name = 'other'

print(sd.query_devices())
sample_rate = 16000
chains = 8
full_block_len = 1000 #msec
block_duration = round(full_block_len/chains)
block_size = round(sample_rate * block_duration / 1000)
device = 2 #sysdefault

samples = np.zeros(round(sample_rate * full_block_len / 1000))

wait_samples = 0

SAMPLES_TO_WAIT = 2
SAMPLES_TO_END  = 3

STATE_IDLE = 0
STATE_WAIT = 1
STATE_REC  = 2
STATE_END  = 3
state = STATE_IDLE

VAL_MAX_TRIGGER = 0.01

c = None
index = 0

try:
    def callback(indata, frames, time, status):
        global samples
        global state
        global wait_samples

        if status:
            text = '-- ' + str(status) + ' --'
            print(text)
        if any(indata):
            samples=np.roll(samples, -block_size)
            samples[-block_size-1:-1] = indata[:, 0]
            #sd.play(samples,sample_rate)

            maximum = np.max(indata[:, 0])
            #print('max %f' % (maximum))

            if   state == STATE_IDLE:
                if maximum > VAL_MAX_TRIGGER:
                    state = STATE_WAIT
                    wait_samples = 0
            elif state == STATE_WAIT:
                #print('wait')
                wait_samples += 1
                if wait_samples > SAMPLES_TO_WAIT:
                    state = STATE_REC
            elif state == STATE_REC:
                #print('rec')

                q.put(samples)

                state = STATE_END
                wait_samples = 0
            elif state == STATE_END:
                #print('wait end')
                wait_samples += 1
                if wait_samples > SAMPLES_TO_END:
                    #print('idle')
                    state = STATE_IDLE
            else:
                state = STATE_IDLE

        else:
            print('no input')

    with sd.InputStream(device=device, channels=1, callback=callback,
                        blocksize=int(sample_rate * block_duration / 1000),
                        samplerate=sample_rate):
        while True:
            get_samples = q.get()
            if get_samples is None:
                continue

            #process_sound_save(get_samples)
            spectrogram = get_sound_spectrogram(get_samples)
            process_sound_predict(spectrogram)
            process_sound_plot(spectrogram)

except KeyboardInterrupt:
    pass
except Exception as e:
    exit(type(e).__name__ + ': ' + str(e))
