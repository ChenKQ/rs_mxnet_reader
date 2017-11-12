import mxnet as mx
import os,sys
from mxnet.io import DataIter
import numpy as np
import random
from GdalReader import labelViz,GdalReader

class GdalSampleStore(object):
    '''
    It is a class that stores samples. Each sample contanins sevel topgdalreaders and one gtgdalreader.
    '''
    def __init__(self,lstfile,isTrain=True,rgbMeans=[],data_root=''):
        '''
        Initializtion Method. lstfile is list file. rgbMeans shuold be a list that contains sevel list, each of which is mean values of each channel.
        :param lstfile: a file that lists all training samples
        :param isTrain:  True or Fals. If True, its gt should be provided.
        :param rgbMeans: [[rgb_mean of dsm],[rgb_mean of rgb]]
        '''
        self.lstfile=lstfile
        self.isTrain=isTrain
        self.rgbMeans=rgbMeans
        self.data_root=data_root
        self.samples=[]
        self.parseLstfile()
    def parseLstfile(self):
        '''
        parse list file and save the samples in self.samples. Each sample contains several readers.
        :return:
        '''
        with open(self.lstfile,'r') as f:
            lines=f.readlines()
        for line in lines:
            sample=self.parseLine(line)
            self.samples.append(sample)
    def parseLine(self,line):
        '''
        parse each line in list file. Each line can be converted to several readers. Each image corresponds to a reader.
        :param line: string of each line. Differnt image are splited with space and the last image should be ground truth is self.isTrain is True.
        :return:
        '''
        sample={}
        files = line.strip('\n').strip(' ').split(' ')
        lenfiles=len(files)
        nchannel=0
        if not self.isTrain:
            topgdalreaders=[]
            gtgdalreaders=[]
            for idx,f in enumerate(files):
                topgdalreaders.append(GdalReader(os.path.join(self.data_root,f),rgbMean=self.rgbMeans[idx],colorMap=True))
                nchannel += topgdalreaders[idx].dataset.RasterCount
            sample['top'] = topgdalreaders
            sample['gt'] = gtgdalreaders
            sample['nchannel']=nchannel
        else:
            topgdalreaders=[]
            gtgdalreaders=[]
            for idx,f in enumerate(files):
                if idx==(lenfiles-1):
                    gtgdalreaders.append(GdalReader(os.path.join(self.data_root,f),rgbMean=[],colorMap=False))
                else:
                    topgdalreaders.append(GdalReader(os.path.join(self.data_root,f),rgbMean=self.rgbMeans[idx],colorMap=True))
                    nchannel += topgdalreaders[idx].dataset.RasterCount
            sample['top'] = topgdalreaders
            sample['gt'] = gtgdalreaders
            sample['nchannel'] = nchannel
        return sample
    def getOneSample(self):
        '''
        From all the samples, return one sample
        :return:
        '''
        return random.sample(self.samples,1)[0]

    def getAnySample(self,no):
        '''
        From all samples, return #no samples.
        :param no:
        :return:
        '''
        samples=[]
        for i in range(no):
            samples.append(self.getOneSample())
        return samples


class SimpleBatch(object):
    def __init__(self,data,label):
        self.data=data
        self.label=label

class RSReader(DataIter):
    '''
    It is a iterator that can be feed into deep networks. It is designed for remote sensing scene and parse images with gdal.
    '''
    def __init__(self,flist_name,batch_size,epochiter=2000,div=False,data_root='',
                 rgbMeans=[[]],cutoff_size=224,isTrain=True,DataAug=False,isMirror=False,isRotate=False,
                 data_name='data',
                 label_name='softmax_label'):
        super(DataIter,self).__init__()
        self.flist_name=os.path.join(data_root,flist_name)
        self.batch_size=batch_size
        self.div=div
        self.data_root=data_root
        self.rgbMeans=rgbMeans
        self.cutoff_size=cutoff_size
        self.isTrain = isTrain
        self.DataAug = DataAug
        self.isMirror = isMirror
        self.isRotate = isRotate
        self.data_name=data_name
        self.label_name=label_name
        self.epochiter=epochiter
        self.sampleStore=GdalSampleStore(self.flist_name,isTrain=self.isTrain,rgbMeans=self.rgbMeans,data_root=data_root)
        self.data,self.label=self._read()
        self.cursor= 0

    def _randomBool(self):
        xx=[True,False]
        return random.sample(xx,1)[0]

    def _randomRotate(self):
        xx=[0,1,2,3]
        return random.sample(xx,1)[0]

    def _sampleOne(length):
        return random.sample(range(length), 1)[0]

    def _randomCrop(self,patchsize):
        if not self.cutoff_size is None:
            return round(np.random.rand()*(patchsize-self.cutoff_size))

    def _read(self):
        data={}
        label={}
        data[self.data_name]=[]
        label[self.label_name]=[]
        samples=self.sampleStore.getAnySample(self.batch_size)
        # print sys.getsizeof(samples)
        for sample in samples:
            data_,label_=self._read_img(sample)
            data[self.data_name].append(data_)
            label[self.label_name].append(label_)
        data[self.data_name]=np.asarray(data[self.data_name])
        label[self.label_name]=np.asarray(label[self.label_name])
        return data.items(),label.items()

    def _read_img(self,sample,initx=None,inity=None):
        # if sample is None:
        #     sample = self.sampleStore.samples[0]
        imgReaders=sample['top']
        if self.isTrain:
            gtReader=sample['gt'][0]
        else:
            gtReader = None
        nchannel= sample['nchannel']
        img=np.zeros((nchannel,self.cutoff_size,self.cutoff_size),dtype=np.float64)
        if initx is None:
            initx = self._randomCrop(imgReaders[0].dataset.RasterXSize)
        if inity is None:
            inity = self._randomCrop(imgReaders[0].dataset.RasterYSize)
        startchannel = 0
        for idx,reader in enumerate(imgReaders):
            img[startchannel:startchannel+reader.dataset.RasterCount,...]=reader.readPatchAsNumpy(initx,inity,self.cutoff_size,height=self.cutoff_size,
                                                                                                  bandlst=[],dtype=np.float64,hwcorder=False,converter=None)
            startchannel += reader.dataset.RasterCount
        if self.isTrain:
            label=gtReader.readPatchAsNumpy(initx,inity,self.cutoff_size,height=self.cutoff_size,bandlst=[],dtype=np.uint8,hwcorder=False,converter=None)
            assert img.shape[1:] == label.shape
            if self.div:
                label = np.array(label) / 255
            else:
                label = np.array(label)
        else:
            label=None
        if self.isTrain and self.DataAug:
            # #flip
            if self.isMirror:
                isHorizonFlip=self._randomBool()
                isVerticalFlip=self._randomBool()
                if isHorizonFlip:
                    img=img[:,::-1,:]
                    label=label[::-1,:]
                if isVerticalFlip:
                    img = img[:,:,::-1]
                    label=label[:,::-1]
            #rotate
            rotateAngle=self._randomRotate()
            if self.isRotate and not rotateAngle==0:
                for i in range(img.shape[0]):
                    img[i,...] =np.rot90(img[i,...],rotateAngle)
                label=np.rot90(label,rotateAngle)
        return (img,label)

    def read_img(self,sample,initx,inity):
        return self._read_img(sample,initx,inity)

    @property
    def provide_data(self):
        return [(k,tuple([self.batch_size] + list(v.shape[1:]))) for k,v in self.data]

    @property
    def provide_label(self):
        return [(k,tuple([self.batch_size] + list(v.shape[1:]))) for k,v in self.label]

    def get_batch_size(self):
        return self.batch_size

    def reset(self):
        self.cursor = 0

    def iter_next(self):
        self.cursor += 1
        if self.cursor<self.epochiter:
            return True
        else:
            return False

    def next(self):
        if self.iter_next():
            self.data,self.label=self._read()
            if self.isTrain:
                return SimpleBatch([mx.nd.array(self.data[0][1])],[mx.nd.array(self.label[0][1])])
            else:
                return SimpleBatch([mx.nd.array(self.data[0][1])], None)
        else:
            raise StopIteration

def calMean():
    train_flist_name='/home/yanml/mxtest/semantic/Data/vaihingen/trainset.lst'
    val_flist_name='/home/yanml/mxtest/semantic/Data/vaihingen/valset.lst'
    test_flist_name='/home/yanml/mxtest/semantic/Data/vaihingen/testset.lst'
    batch_size=1
    trainreader = RSReader(train_flist_name,batch_size,epochiter=20000,div=False,
                 rgbMeans=[[],[]],cutoff_size=224,isTrain=True,DataAug=False,isMirror=False,isRotate=False,
                 data_name='data',
                 label_name='softmax_label')
    valreader=RSReader(val_flist_name,batch_size,epochiter=10000,div=False,
                 rgbMeans=[[],[]],cutoff_size=224,isTrain=True,DataAug=False,isMirror=False,isRotate=False,
                 data_name='data',
                 label_name='softmax_label')
    testreader=RSReader(test_flist_name,batch_size,epochiter=200000,div=False,
                 rgbMeans=[[],[]],cutoff_size=224,isTrain=False,DataAug=False,isMirror=False,isRotate=False,
                 data_name='data',
                 label_name='softmax_label')
    count =0
    import time
    t1=time.time()
    sum = np.zeros((4,))
    count = 0L
    converter=labelViz('../DataPrepare/isprs/isprs.txt')
    from matplotlib import pyplot as plt
    for bat in trainreader:
        img=bat.data
        label=bat.label
        imgshow=img[0].asnumpy()[0,1:4]
        imgshow=np.swapaxes(imgshow,0,2)
        imgshow=np.swapaxes(imgshow,0,1)
        imgshow=np.asarray(imgshow,dtype=np.uint8)
        plt.subplot(121)
        plt.imshow(imgshow)
        plt.subplot(122)
        plt.imshow(converter.Viz(np.asarray(label[0].asnumpy()[0],dtype=np.uint8)))
        sum += np.sum(img[0].asnumpy()[0,...],axis=(1,2))
        count += img[0].shape[2]*img[0].shape[3]
        # print type(img)
        # print type(label)
    print sum/count
    t2=time.time()
    print count,t2-t1


if __name__=='__main__':
    calMean()