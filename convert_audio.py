import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import scipy.misc
import pathlib
from tifffile import imsave
import random
import os, shutil

prefix = os.getcwd()
directory = prefix + '/speech_commands_v0.01'

validation_part = 0.1
test_part = 0.01
folder_base     = prefix+'/spectrograms'
folder_on       = '/on'
folder_off      = '/off'
folder_other    = '/other'
folder_validation  = '/validate'
folder_test        = '/test'
folder_train       = '/train'
# ---------------- Remove old files ----------------
print('Remove files in %s..'%(folder_base))
yes = input("Proceed? y/n")
if yes!='y':
    exit(0)
print('Remove..')
for filename in os.listdir(folder_base):
    file_path = os.path.join(folder_base, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

os.makedirs(folder_base+folder_train+folder_on, exist_ok=True)
os.makedirs(folder_base+folder_validation+folder_on, exist_ok=True)
os.makedirs(folder_base+folder_test+folder_on, exist_ok=True)
os.makedirs(folder_base+folder_train+folder_off, exist_ok=True)
os.makedirs(folder_base+folder_validation+folder_off, exist_ok=True)
os.makedirs(folder_base+folder_test+folder_off, exist_ok=True)
os.makedirs(folder_base+folder_train+folder_other, exist_ok=True)
os.makedirs(folder_base+folder_validation+folder_other, exist_ok=True)
os.makedirs(folder_base+folder_test+folder_other, exist_ok=True)
# ---------------- Add new files ----------------
global mic_noise
sample_rate, mic_noise = wavfile.read("mic_noise.wav")
maximum = np.max(mic_noise)
mic_noise = mic_noise / maximum

def add_mic_noise(ssamples):
    global mic_noise
    insamples=samples
    nstart = np.random.randint(len(mic_noise)-len(insamples))
    nampl  = 5*random.random()
    for i in range(len(insamples)):
        insamples[i] = insamples[i] + mic_noise[i+nstart]*nampl
    return insamples

def rotate(insamples):
    L=len(insamples)*0.3
    n = int(np.random.randint(L)-L/2)
    return np.roll(insamples, n)

def reverse(samples):
    samples.reverse()
    return samples

for root, subdirs, files in os.walk(directory):
#for filename in os.listdir(directory):
    for filename in files:
        if filename.endswith(".wav"):
            choice = random.random()
            out_path = folder_base
            if choice < test_part:
                out_path += folder_test
            elif choice < validation_part:
                out_path += folder_validation
            else:
                out_path += folder_train

            oth = False
            if (root.endswith('/on')):
                out_path += folder_on
            elif (root.endswith('/off')):
                out_path += folder_off
            else:
                oth = True
                out_path += folder_other

            full_filename = os.path.join(root, filename)
            print('Read %s'%(full_filename))
            sample_rate, samples = wavfile.read(full_filename)
            #maximum = np.max(samples)
            #samples = samples/maximum

            for k in range(8):
                if oth and (random.random() > (1/15)):
                    continue
                #samples_n = add_mic_noise(samples)
                samples_n = rotate(samples)

                #print('sample_rate %d samples %d'%(sample_rate, len(samples)))
                # frequencies size = nperseg/2 + 1
                # times size = len(samples)/nperseg if noverlap == 0
                frequencies, times, spectrogram = signal.spectrogram(samples_n, sample_rate, nperseg=256, noverlap = 0)
                maximum = np.max(spectrogram)
                spectrogram = spectrogram/maximum

                if 0:
                    plt.pcolormesh(times, frequencies, spectrogram)
                    #plt.imshow(spectrogram, aspect=len(times)/len(frequencies))
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [sec]')
                    plt.show()
                    #exit(1)

                pure_filename = str(pathlib.Path(full_filename).stem) + '_' + str(k)
                pure_filename = pure_filename + '.tif'
                full_filename_img = os.path.join(out_path, pure_filename)

                imsave(full_filename_img, spectrogram.astype(np.float32))

            #break

