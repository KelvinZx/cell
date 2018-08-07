import numpy as np
import scipy.misc as misc
import os
import glob
import matplotlib.pyplot as plt

class CRC_joint:

    def __init__(self,imgdir,target_size=256):
        self.imgdir = imgdir
        self.target_size = target_size

    def load_train(self, preprocess=True):
        x_train = []
        y_train_det = []
        y_train_cls = []
        for i in range(1,71):
            for k in range(1,7):
                img_file_name = 'img' + str(i) + '_' + str(k) + '.bmp'
                detection_file_name = 'img' + str(i) + '_' + str(k) + '_detectionL.bmp'
                classification_file_name = 'img' + str(i) + '_' + str(k) + '_ClassificationL.bmp'
                img_path = os.path.join(self.imgdir, img_file_name)
                detection_path = os.path.join(self.imgdir, detection_file_name)
                classification_path = os.path.join(self.imgdir, classification_file_name)
                if os.path.exists(img_path):
                    img = misc.imread(img_path)
                    img = misc.imresize(img,(self.target_size,self.target_size),interp='nearest')
                    if preprocess:
                        # img = img/255.
                        #img -= np.mean(img, keepdims=True)
                        #img /= (np.std(img, keepdims=True) + 1e-7)
                        img = img - 128.
                        img = img / 128.
                    x_train.append(img)

                    detection = misc.imread(detection_path, mode='L')
                    detection = misc.imresize(detection, (self.target_size, self.target_size), interp='nearest')

                    classification = misc.imread(classification_path, mode='L')
                    classification = misc.imresize(classification, (self.target_size, self.target_size), interp='nearest')

                    detection = detection.reshape(detection.shape[0], detection.shape[1], 1)
                    classification = classification.reshape(classification.shape[0], classification.shape[1], 1)

                    y_train_det.append(detection)
                    y_train_cls.append(classification)

            print('finsihed:img',i)
        x_train = np.array(x_train)
        x_train = np.transpose(x_train, (0, 3, 1, 2))
        y_train_det = np.array(y_train_det)
        y_train_det = np.transpose(y_train_det, (0, 3, 1, 2))
        y_train_cls = np.array(y_train_cls)
        y_train_cls = np.transpose(y_train_cls, (0, 3, 1, 2))

        return x_train, y_train_det, y_train_cls

    def load_val(self, preprocess=True):
        x_train = []
        y_train_det = []
        y_train_cls = []
        for i in range(71, 81):
            for k in range(1, 7):
                img_file_name = 'img' + str(i) + '_' + str(k) + '.bmp'
                detection_file_name = 'img' + str(i) + '_' + str(k) + '_detectionL.bmp'
                classification_file_name = 'img' + str(i) + '_' + str(k) + '_ClassificationL.bmp'
                img_path = os.path.join(self.imgdir, img_file_name)
                detection_path = os.path.join(self.imgdir, detection_file_name)
                classification_path = os.path.join(self.imgdir, classification_file_name)
                if os.path.exists(img_path):
                    img = misc.imread(img_path)
                    img = misc.imresize(img, (self.target_size, self.target_size), interp='nearest')
                    if preprocess:
                        # img = img/255.
                        # img -= np.mean(img, keepdims=True)
                        # img /= (np.std(img, keepdims=True) + 1e-7)
                        img = img - 128.
                        img = img / 128.
                    x_train.append(img)

                    detection = misc.imread(detection_path, mode='L')
                    detection = misc.imresize(detection, (self.target_size, self.target_size), interp='nearest')

                    classification = misc.imread(classification_path, mode='L')
                    classification = misc.imresize(classification, (self.target_size, self.target_size),
                                                   interp='nearest')

                    detection = detection.reshape(detection.shape[0], detection.shape[1], 1)
                    classification = classification.reshape(classification.shape[0], classification.shape[1], 1)

                    y_train_det.append(detection)
                    y_train_cls.append(classification)

            print('finsihed:img', i)
        x_train = np.array(x_train)
        x_train = np.transpose(x_train, (0, 3, 1, 2))
        y_train_det = np.array(y_train_det)
        y_train_det = np.transpose(y_train_det, (0, 3, 1, 2))
        y_train_cls = np.array(y_train_cls)
        y_train_cls = np.transpose(y_train_cls, (0, 3, 1, 2))

        return x_train, y_train_det, y_train_cls


