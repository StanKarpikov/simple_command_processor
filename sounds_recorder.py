from scipy import signal
from scipy.io import wavfile
import numpy as np
import sounddevice as sd
import queue
import os
import matplotlib.pyplot as plt
import matplotlib
import time

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

VAL_MAX_TRIGGER = 0.1

c = None
index = 0

try:
    def callback(indata, frames, time, status):
        global samples
        global state
        global wait_samples
        global c

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
            timestr = time.strftime("%Y%m%d_%H%M%S")
            dir = directory + '/' +  sound_name + '/'
            filename = timestr + '_' + str(index) + '.wav'
            full_filename = dir + filename
            os.makedirs(dir, exist_ok=True)
            wavfile.write(full_filename, sample_rate, get_samples)
            print('%s written'%(filename))
            index += 1

            if True:
                frequencies, times, spectrogram = signal.spectrogram(get_samples, sample_rate, nperseg=256, noverlap=0)
                maximum = np.max(spectrogram)
                spectrogram = np.array(spectrogram / maximum)

                times = np.linspace(start=0, stop=1, num=input_size_x)
                frequencies = np.linspace(start=0, stop=8000, num=input_size_y)
                if c is not None:
                    c.remove()
                c = plt.pcolormesh(times, frequencies, spectrogram)
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plot_pause(0.1)

except KeyboardInterrupt:
    pass
except Exception as e:
    exit(type(e).__name__ + ': ' + str(e))