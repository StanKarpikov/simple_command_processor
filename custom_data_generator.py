import numpy as np
import cv2
import os
from tensorflow.keras.utils import Sequence
import math
from tensorflow.keras.utils import to_categorical

class DataGenerator(Sequence):

    def __init__(self):
        self.x = []
        self.y = []
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
                            shuffle = False):
        self.batch_size = batch_size
        self.directory = directory
        self.target_size = target_size
        self.color_mode = color_mode
        self.class_mode = class_mode
        self.shuffle = shuffle
        self.next_idx = 0

        self.categories=[]
        self.x = []
        self.y = []
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
                                self.x.append(file_path)
                                self.y.append(cat_num)
                                files_num += 1
                        except Exception as e:
                            print('Failed to load %s. Reason: %s' % (file_path, e))
                    print('Load category: %s files %d'%(category, files_num))
            except Exception as e:
                print('Failed to load %s. Reason: %s' % (file_path, e))
            cat_num += 1
        return self

    def get_categories(self):
        return self.categories

    def __len__(self):
        # Number of bunches per epoch
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        #y_binary = [to_categorical(curr_y) for curr_y in batch_y]

        return np.array([self._load_image(file_name) for file_name in batch_x]), np.array(batch_y)

    def next(self):
        batch_x = self.x[self.next_idx * self.batch_size:(self.next_idx + 1) * self.batch_size]
        batch_y = self.y[self.next_idx * self.batch_size:(self.next_idx + 1) * self.batch_size]
        self.next_idx += 1
        return np.array([self._load_image(file_name) for file_name in batch_x]), np.array(batch_y)

    def _load_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path, -1)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        return img