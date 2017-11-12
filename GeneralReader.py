import mxnet as mx
import os,sys
from mxnet.io import DataIter
import numpy as np
import random
from GdalReader import GdalReader
from OpenCVReader import OpenCVReader
from SimpleBatch import SimpleBatch

class SampleStore(object):
    '''
    It is a class that stores samples. Each sample contanins sevel topgdalreaders and one gtgdalreader.
    '''
    def __init__(self,lstfile,isTrain=True,rgbMeans=None,data_root='',withGdal=True):
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
        self.withGdal = withGdal
        self.openlimits = 50
        self.openAtFirst = False
        self.samples=[]
        self.parseLstfile()
    def parseLstfile(self):
        '''
        parse list file and save the samples in self.samples. Each sample contains several readers.
        :return:
        '''
        with open(self.lstfile,'r') as f:
            lines=f.readlines()
        if len(lines) >=self.openlimits:
            self.openAtFirst = False
        else:
            self.openAtFirst = True
        for line in lines:
            if self.openAtFirst and self.withGdal:
                sample = self.parseLine2Reader(line)
            else:
                sample = self.parseLine2Str(line)
            self.samples.append(sample)

    def parseLine2Str(self,line):
        sample={}
        files = line.strip('\n').strip(' ').split(' ')
        lenfiles=len(files)
        if not self.isTrain:
            topfiles=[]
            gtfiles=[]
            for idx,f in enumerate(files):
                topfiles.append(os.path.join(self.data_root,f))
            sample['top'] = topfiles
            sample['gt'] = gtfiles
        else:
            topfiles=[]
            gtfiles=[]
            for idx,f in enumerate(files):
                if idx==(lenfiles-1):
                    gtfiles.append(os.path.join(self.data_root,f))
                else:
                    topfiles.append(os.path.join(self.data_root,f))
            sample['top'] = topfiles
            sample['gt'] = gtfiles
        return sample

    def parseLine2Reader(self,line):
        '''
        parse each line in list file. Each line can be converted to several readers. Each image corresponds to a reader.
        :param line: string of each line. Differnt image are splited with space and the last image should be ground truth is self.isTrain is True.
        :return:
        '''
        assert self.withGdal and self.openAtFirst
        sample={}
        files = line.strip('\n').strip(' ').split(' ')
        lenfiles=len(files)
        nchannel=0
        if not self.isTrain:
            topgdalreaders=[]
            gtgdalreaders=[]
            for idx,f in enumerate(files):
                if self.rgbMeans is None:
                    topgdalreaders.append(GdalReader(os.path.join(self.data_root, f), rgbMean=None, colorMap=True))
                else:
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
                    if not self.rgbMeans is None:
                        topgdalreaders.append(GdalReader(os.path.join(self.data_root,f),rgbMean=self.rgbMeans[idx],colorMap=True))
                    else:
                        topgdalreaders.append(GdalReader(os.path.join(self.data_root, f), rgbMean=None, colorMap=True))
                    nchannel += topgdalreaders[idx].dataset.RasterCount
            sample['top'] = topgdalreaders
            sample['gt'] = gtgdalreaders
            sample['nchannel'] = nchannel
        return sample

    def getOneSample(self,idx=-1):
        '''
        From all the samples, return one sample
        :return:
        '''
        if self.openAtFirst and self.withGdal:
            if idx == -1:
                return random.sample(self.samples,1)[0]
            else:
                return self.samples[idx]
        elif self.withGdal:
            sample = {}
            if idx ==-1:
                filesample = random.sample(self.samples,1)[0]
            else:
                filesample = self.samples[idx]
            topfiles = filesample['top']
            gtfiles = filesample['gt']
            topreaders = []
            gtreaders = []
            nchannel = 0
            for idx,f in enumerate(topfiles):
                if self.rgbMeans is None:
                    topreaders.append(GdalReader(f, rgbMean=None, colorMap=True))
                else:
                    topreaders.append(GdalReader(f,rgbMean=self.rgbMeans[idx],colorMap=True))
                nchannel += topreaders[idx].dataset.RasterCount
            for idx,f in enumerate(gtfiles):
                gtreaders.append(GdalReader(f,rgbMean=None,colorMap=False))
            sample['top'] = topreaders
            sample['gt'] = gtreaders
            sample['nchannel'] = nchannel
            return sample
        else:
            sample = {}
            if idx==-1:
                filesample = random.sample(self.samples,1)[0]
            else:
                filesample = self.samples[idx]
            topfiles = filesample['top']
            gtfiles = filesample['gt']
            topreaders = []
            gtreaders = []
            nchannel = 0
            for idx,f in enumerate(topfiles):
                if self.rgbMeans is None:
                    topreaders.append(OpenCVReader(f, rgbMean=None, colorMap=True))
                else:
                    topreaders.append(OpenCVReader(f,rgbMean=self.rgbMeans[idx],colorMap=True))
                nchannel += topreaders[idx].getNChannel()
            for idx,f in enumerate(gtfiles):
                gtreaders.append(OpenCVReader(f,rgbMean=self.rgbMeans[idx],colorMap=False))
            sample['top'] = topreaders
            sample['gt'] = gtreaders
            sample['nchannel'] = nchannel
            return sample

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


class GeneralReader(DataIter):
    '''
    It is a iterator that can be feed into deep networks. It is designed for remote sensing scene and parse images with gdal.
    '''
    def __init__(self,flist_name,batch_size,epochiter=2000,div=False,data_root='',withGdal=True,
                 rgbMeans=None,bandlist=None,cutoff_size=224,isTrain=True,DataAug=None,
                 data_name='data',
                 label_name='softmax_label'):
        super(DataIter,self).__init__()
        self.flist_name=os.path.join(data_root,flist_name)
        self.batch_size=batch_size
        self.div=div
        self.data_root=data_root
        self.withGdal = withGdal
        self.rgbMeans=rgbMeans
        self.bandlist = bandlist
        self.cutoff_size=cutoff_size
        self.isTrain = isTrain
        self.DataAug = DataAug
        self.data_name=data_name
        self.label_name=label_name
        self.epochiter=epochiter
        self.sampleStore=SampleStore(self.flist_name,isTrain=self.isTrain,rgbMeans=self.rgbMeans,data_root=data_root,withGdal=self.withGdal)
        self.data,self.label=self._read()
        self.cursor= 0

    # def _randomBool(self):
    #     xx=[True,False]
    #     return random.sample(xx,1)[0]

    # def _randomRotate(self):
    #     xx=[0,1,2,3]
    #     return random.sample(xx,1)[0]

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
        for i in range(self.batch_size):
            sample = self.sampleStore.getOneSample()
            data_,label_=self._read_img(sample)
            data[self.data_name].append(data_)
            if not label_ is None:
                label[self.label_name].append(label_)
        data[self.data_name]=np.asarray(data[self.data_name])
        if len(label[self.label_name]) == 0:
            label[self.label_name]=None
        else:
            label[self.label_name]=np.asarray(label[self.label_name])
        return data.items(),label.items()   #('data',np.array) , ('softmax',np.array) shape: (batchsize,channel,height,width)

    def read(self,sampleidx,initx,inity):
        data={}
        label={}
        data[self.data_name]=[]
        label[self.label_name]=[]
        for i in range(self.batch_size):
            sample = self.sampleStore.samples[sampleidx]
            data_,label_=self._read_img(sample,initx,inity)
            data[self.data_name].append(data_)
            label[self.label_name].append(label_)
        data[self.data_name]=np.asarray(data[self.data_name])
        label[self.label_name]=np.asarray(label[self.label_name])
        return data.items(),label.items()   #('data',np.array) , ('softmax',np.array) shape: (batchsize,channel,height,width)

    def _read_img(self,sample,initx=None,inity=None):
        '''
        This method only read one patch from one sample. One sample is an instance of SimpleBatch
        :param sample: an instance of SimpleBatch
        :param initx:
        :param inity:
        :return:
        '''
        # if sample is None:
        #     sample = self.sampleStore.samples[0]
        imgReaders=sample['top']
        if self.isTrain:
            gtReader=sample['gt'][0]
        else:
            gtReader = None
        if self.bandlist is None:
            nchannel= sample['nchannel']
        else:
            nchannel = 0
            for cc in self.bandlist:
                nchannel += len(cc)
        img=np.zeros((nchannel,self.cutoff_size,self.cutoff_size),dtype=np.float64)
        size = imgReaders[0].getSize()
        if initx is None:
            initx = self._randomCrop(size[1])
        if inity is None:
            inity = self._randomCrop(size[0])
        startchannel = 0
        for idx,reader in enumerate(imgReaders):
            if self.bandlist is None:
                nchannel = reader.getNChannel()
                img[startchannel:startchannel + nchannel, ...] = reader.readPatchAsNumpy(initx, inity, self.cutoff_size,
                                                                                         height=self.cutoff_size,
                                                                                         bandlst=None,
                                                                                         dtype=np.float64,
                                                                                         hwcorder=False, converter=None)
            else:
                nchannel = len(self.bandlist[idx])
                img[startchannel:startchannel+nchannel,...]=reader.readPatchAsNumpy(initx,inity,self.cutoff_size,height=self.cutoff_size,
                                                                                                  bandlst=self.bandlist[idx],dtype=np.float64,hwcorder=False,converter=None)
            startchannel += nchannel
        if self.isTrain:
            label=gtReader.readPatchAsNumpy(initx,inity,self.cutoff_size,height=self.cutoff_size,bandlst=[],dtype=np.uint8,hwcorder=False,converter=None)
            assert img.shape[1:] == label.shape
            if self.div:
                label = np.array(label) / 255
            else:
                label = np.array(label)
        else:
            label=None
        if not self.DataAug is None and not self.DataAug is False:
            img,label = self.DataAug.aug(img,label)
        return (img,label)

    def read_img(self,sample,initx,inity):
        return self._read_img(sample,initx,inity)

    @property
    def provide_data(self):
        return [(k,tuple([self.batch_size] + list(v.shape[1:]))) for k,v in self.data]

    @property
    def provide_label(self):
        if self.label[0][1]==None:
            return None
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

if __name__ =='__main__':
    trainlst_name = '/mnt/chenkq/data/vaihingen/trainset.lst'
    vallst_name = '/mnt/chenkq/data/vaihingen/valset.lst'
    data_root = '/mnt/chenkq/data/vaihingen'
    net_prefix = '/mnt/chenkq/models/NIdea/vaihingen_deconv_weight'
    rgb_mean = [[283.43644364], [120.47595769, 81.79931481, 81.19268267], [0]]
    cropsize = 224
    batchsize = 16
    trainepochiter = 500
    reader = GeneralReader(trainlst_name, batchsize, epochiter=trainepochiter, div=False, data_root=data_root,
                           withGdal=True,
                           rgbMeans=rgb_mean, cutoff_size=cropsize, isTrain=False, DataAug=False, isMirror=False,
                           isRotate=False)