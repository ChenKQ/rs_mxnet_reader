import mxnet as mx
from mxnet.io import DataIter
import numpy as np
import random
import cv2
import os

class SimpleBatch(object):
    def __init__(self,data,label):
        self.data=data
        self.label=label


class SegReader(DataIter):
    '''
    It is the third edition of data reader. It adopts SimpleBatch to adopt to Module class.
    It works only when the images are not geotiff.
    '''
    def __init__(self,root_dir,flist_name,batch_size,div=True,
                 rgb_mean=(105.229,105.247,97.835),cutoff_size=None,isTrain=True,DataAug=False,isMirror=False,isRotate=False,
                 data_name='data',
                 label_name='softmax_label'):
        super(DataIter,self).__init__()
        self.root_dir=root_dir
        self.flist_name=flist_name
        self.batch_size=batch_size
        self.div=div
        self.mean=np.array(rgb_mean)
        self.cutoff_size=cutoff_size
        self.isTrain = isTrain
        self.DataAug = DataAug
        self.isMirror = isMirror
        self.isRotate = isRotate
        self.data_name=data_name
        self.label_name=label_name
        self.num_data=len(open(self.flist_name,'r').readlines())
        self.f= open(self.flist_name,'r')
        self.data,self.label=self._read()
        self.cursor=-1

    def _randomBool(self):
        xx=[True,False]
        return random.sample(xx,1)[0]

    def _randomRotate(self):
        xx=[0,1,2,3]
        return random.sample(xx,1)[0]

    def _randomCrop(self,patchsize):
        if not self.cutoff_size is None:
            return int(np.random.rand()*(patchsize-self.cutoff_size))


    def _read(self):
        data={}
        label={}
        data[self.data_name]=[]
        label[self.label_name]=[]
        for i in range(self.batch_size):
            try:
                data_img_name,label_img_name=self.f.readline().strip('\n').split(' ')
                data_img_name = os.path.join(self.root_dir,data_img_name)
                label_img_name = os.path.join(self.root_dir,label_img_name)
            except:
                x=self.f.readline().strip('\n').split('\t')
                print x
                print len(x)
                return
            data_,label_=self._read_img(data_img_name,label_img_name)
            data[self.data_name].append(data_)
            label[self.label_name].append(label_)
        data[self.data_name]=np.asarray(data[self.data_name])
        label[self.label_name]=np.asarray(label[self.label_name])
        return data.items(),label.items()

    def _read_img(self,img_name,label_name):
        # img_pil=Image.open(img_name)
        img_pil=cv2.imread(img_name,-1)
        # label_pil=Image.open(label_name)
        label_pil=cv2.imread(label_name,-1)
        # assert img_pil.size==label_pil.size
        assert img_pil.shape[0:2]==label_pil.shape
        img_ndarray=np.array(img_pil,dtype=np.float32)
        if self.div:
            label_ndarray=np.array(label_pil)/255
        else:
            label_ndarray=np.array(label_pil)
        # if self.cutoff_size is not None:
        #     pass
        # print img_ndarray.shape,label_ndarray.shape
        if self.isTrain and self.DataAug:
            initx=self._randomCrop(label_ndarray.shape[0])
            inity=self._randomCrop(label_ndarray.shape[1])
            #crop to destination shape
            img_ndarray = img_ndarray[initx:initx+self.cutoff_size,inity:inity+self.cutoff_size,:]
            label_ndarray=label_ndarray[initx:initx+self.cutoff_size,inity:inity+self.cutoff_size]
            # #flip
            if self.isMirror:
                isHorizonFlip=self._randomBool()
                isVerticalFlip=self._randomBool()
                if isHorizonFlip:
                    img_ndarray=img_ndarray[::-1,:,:]
                    label_ndarray=label_ndarray[::-1,:]
                if isVerticalFlip:
                    img_ndarray = img_ndarray[:,::-1,:]
                    label_ndarray=label_ndarray[:,::-1]
            #rotate
            rotateAngle=self._randomRotate()
            if self.isRotate and not rotateAngle==0:
                img_ndarray =np.rot90(img_ndarray,rotateAngle)
                label_ndarray=np.rot90(label_ndarray,rotateAngle)
        # print img_ndarray.shape,label_ndarray.shape
        # cv2.imwrite('/mnt/CKQ/testReader.jpg',img_ndarray)
        # cv2.imwrite('/mnt/CKQ/testReader.png',label_ndarray*255)
        reshaped_mean = self.mean.reshape(1,1,3)
        img_ndarray -= reshaped_mean   #(h,w,c)
        img_ndarray = np.swapaxes(img_ndarray,0,2)  #(c,w,h)
        img_ndarray = np.swapaxes(img_ndarray,1,2)  #(c,h,w)
        #img_ndarray = np.expand_dims(img_ndarray,axis=0)   #(1,c,h,w)
        #label_ndarray = np.expand_dims(label_ndarray,axis=0)  #(1,h,w)
        return (img_ndarray,label_ndarray)

    @property
    def provide_data(self):  #  [('data',(batch,channel,height,width))]
        return [(k,tuple([self.batch_size] + list(v.shape[1:]))) for k,v in self.data]

    @property
    def provide_label(self):
        return [(k,tuple([self.batch_size] + list(v.shape[1:]))) for k,v in self.label]

    def get_batch_size(self):
        return self.batch_size

    def reset(self):
        self.cursor = -1
        self.f.close()
        self.f = open(self.flist_name,'r')

    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor<self.num_data-self.batch_size:
            # print self.num_data,self.cursor
            return True
        # else:
        #     print 'false:',self.cursor
        #     return False

    def next(self):
        if self.iter_next():
            self.data,self.label=self._read()
            return SimpleBatch([mx.nd.array(self.data[0][1])],[mx.nd.array(self.label[0][1])])
        else:
            raise StopIteration


def calMean():
    train_flist_name='/home/yanml/mxtest/semantic/Data/trainset/trainset.lst'
    val_flist_name='/home/yanml/mxtest/semantic/Data/vaihingen/valset.lst'
    test_flist_name='/home/yanml/mxtest/semantic/Data/vaihingen/testset.lst'
    batch_size=1
    trainreader = SegReader(root_dir=None,flist_name=train_flist_name,batch_size=batch_size,div=False,
                 rgb_mean=(0,0,0),cutoff_size=224,isTrain=True,DataAug=False,isMirror=False,isRotate=False,
                 data_name='data',
                 label_name='softmax_label')
    # valreader=SegReader(root_dir=None,flist_name=train_flist_name,batch_size=batch_size,div=False,
    #              rgb_mean=(105.229,105.247,97.835),cutoff_size=None,isTrain=True,DataAug=False,isMirror=False,isRotate=False,
    #              data_name='data',
    #              label_name='softmax_label')
    # testreader=SegReader(root_dir=None,flist_name=train_flist_name,batch_size=batch_size,div=False,
    #              rgb_mean=(105.229,105.247,97.835),cutoff_size=None,isTrain=True,DataAug=False,isMirror=False,isRotate=False,
    #              data_name='data',
    #              label_name='softmax_label')
    count =0
    import time
    t1=time.time()
    sum = np.zeros((3,))
    count = 0L
    for bat in trainreader:
        img=bat.data
        label=bat.label
        sum += np.sum(img[0].asnumpy()[0,...],axis=(1,2))
        count += img[0].shape[2]*img[0].shape[3]
        # print type(img)
        # print type(label)
    print sum/count
    t2=time.time()
    print count,t2-t1


def test():
    trainroot_dir = '/mnt/CKQ/road_extraction/road/trainset'
    trainlst_name = '/mnt/CKQ/roadset/trainset/trainsetlight.lst'
    batchsize=1
    rgb_mean = (116.741, 128.833, 115.812)
    train_iter = SegReader(root_dir=trainroot_dir, batch_size=batchsize, flist_name=trainlst_name, rgb_mean=rgb_mean,
                           cutoff_size=512, isTrain=True, DataAug=True, isMirror=False, isRotate=False)
    stop = 1
    for d in train_iter:
        img = d.data[0].asnumpy()
        gt = d.label[0].asnumpy()
        print 'done'

if __name__=='__main__':
    calMean()
    # test()