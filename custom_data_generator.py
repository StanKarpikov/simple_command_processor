import numpy as np
import cv2
import os
from tensorflow.keras.utils import Sequence
import math
import random
from tensorflow.keras.utils import to_categorical

class DataGenerator(Sequence):

    def __init__(self):
        self.x = []
        self.batch_size = 32
        self.directory = ''
        self.target_size = (100, 100)
        self.color_mode = "grayscale"
        self.class_mode = "categorical"
        self.shuffle = False

    def flow_from_directory(self,
                            directory = '',
                            target_size = (100, 100),
                            color_mode = "grayscale",
                            batch_size = 32,
                            class_mode = "categorical",
                            shuffle_it = False):
        self.batch_size = batch_size
        self.directory = directory
        self.target_size = target_size
        self.color_mode = color_mode
        self.class_mode = class_mode
        self.shuffle_it = shuffle_it
        self.next_idx = 0

        self.categories=[]
        self.x = []
        cat_num = 0
        for dirname in os.listdir(directory):
            dir_path = os.path.join(directory, dirname)
            try:
                if os.path.isdir(dir_path):
                    category = dirname
                    self.categories.append(category)
                    files_num = 0
                    for filename in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                self.x.append([file_path, cat_num])
                                files_num += 1
                        except Exception as e:
                            print('Failed to load %s. Reason: %s' % (file_path, e))
                    print('Load category: %s files %d'%(category, files_num))
            except Exception as e:
                print('Failed to load %s. Reason: %s' % (file_path, e))
            cat_num += 1

        if self.shuffle_it:
            random.shuffle(self.x)

        return self

    def get_categories(self):
        return self.categories

    def __len__(self):
        # Number of bunches per epoch
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array(batch)[:, 0]
        batch_y = np.array(batch)[:, 1]

        #y_binary = [to_categorical(curr_y) for curr_y in batch_y]

        x = np.array([self._load_image(file_name) for file_name in batch_x])

        i=0
        for y in batch_y:
            if y == str(self.categories.index('other')) and np.random.rand()<0.1:
                mx=np.max(x)
                x[i]=np.random.randn(*x[i].shape)*mx
            i+=1

        return x, batch_y

    def _load_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path, -1)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        return img