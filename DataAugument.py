import random
import numpy as np

'''
example:
funcs = [rotate,mirror]
aug = DataAugument(open=True,funcs=funcs)
res = aug.aug(data,label)
'''
class DataAugument(object):
    def __init__(self,open,funcs=[]):
        self.open = open
        self.funcs = funcs

    def aug(self,data,label):
        '''
        rotate image
        :param data: the image to rotate, its shape should be (channel, height, width)
        :param label:
        :return: data,label
        '''
        for f in self.funcs:
            data,label = f(data,label)
        return data,label

def rotate(img,label):
    xx = [0, 1, 2, 3]
    rotateAngle = random.sample(xx, 1)[0]
    for i in range(img.shape[0]):
        img[i, ...] = np.rot90(img[i, ...], rotateAngle)
    label = np.rot90(label, rotateAngle)
    return img,label

def randomBool():
    xx=[True,False]
    return random.sample(xx,1)[0]

def mirror(img,label):
    isHorizonFlip = randomBool()
    isVerticalFlip = randomBool()
    if isHorizonFlip:
        img = img[:, ::-1, :]
        label = label[::-1, :]
    if isVerticalFlip:
        img = img[:, :, ::-1]
        label = label[:, ::-1]
    return img,label

def injuectGaussianNoise(img,label):
    noise = np.random.normal(0.0,1.0,img.shape)
    img += noise
    return img,label

