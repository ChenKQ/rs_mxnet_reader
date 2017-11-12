from GeneralReader import GeneralReader,SampleStore
import os
import numpy as np

class DeepLabReader(GeneralReader):
    def __init__(self,flist_name,batch_size,epochiter=2000,div=False,data_root='',withGdal=True, samplerate=8,
                 rgbMeans=None,bandlist=None,cutoff_size=224,isTrain=True,DataAug=False,
                 data_name='data',
                 label_name='softmax_label'):
        self.samplerate = samplerate
        super(DeepLabReader,self).__init__(flist_name,batch_size,epochiter=epochiter,div=div,data_root=data_root,withGdal=withGdal,
                 rgbMeans=rgbMeans,bandlist=bandlist,cutoff_size=cutoff_size,isTrain=isTrain,DataAug=DataAug,
                 data_name=data_name,
                 label_name=label_name)
        # self.flist_name=os.path.join(data_root,flist_name)
        # self.batch_size=batch_size
        # self.div=div
        # self.data_root=data_root
        # self.withGdal = withGdal
        # self.samplerate = samplerate
        # self.rgbMeans=rgbMeans
        # self.bandlist = bandlist
        # self.cutoff_size=cutoff_size
        # self.isTrain = isTrain
        # self.DataAug = DataAug
        # self.data_name=data_name
        # self.label_name=label_name
        # self.epochiter=epochiter
        # self.sampleStore=SampleStore(self.flist_name,isTrain=self.isTrain,rgbMeans=self.rgbMeans,data_root=data_root,withGdal=self.withGdal)
        # self.data,self.label=self._read()
        # self.cursor= 0

    def _read_img(self,sample,initx=None,inity=None):
        # if sample is None:
        #     sample = self.sampleStore.samples[0]
        # imgReaders=sample['top']
        # if self.isTrain:
        #     gtReader=sample['gt'][0]
        # else:
        #     gtReader = None
        # if self.bandlist is None:
        #     nchannel= sample['nchannel']
        # else:
        #     nchannel = 0
        #     for cc in self.bandlist:
        #         nchannel += len(cc)
        # img=np.zeros((nchannel,self.cutoff_size,self.cutoff_size),dtype=np.float64)
        # size = imgReaders[0].getSize()
        # if initx is None:
        #     initx = self._randomCrop(size[1])
        # if inity is None:
        #     inity = self._randomCrop(size[0])
        # startchannel = 0
        # for idx,reader in enumerate(imgReaders):
        #     if self.bandlist is None:
        #         nchannel = reader.getNChannel()
        #         img[startchannel:startchannel+nchannel,...]=reader.readPatchAsNumpy(initx,inity,self.cutoff_size,height=self.cutoff_size,
        #                                                                                           bandlst=None,dtype=np.float64,hwcorder=False,converter=None)
        #     else:
        #         nchannel = len(self.bandlist[idx])
        #         img[startchannel:startchannel + nchannel, ...] = reader.readPatchAsNumpy(initx, inity, self.cutoff_size,
        #                                                                                  height=self.cutoff_size,
        #                                                                                  bandlst=self.bandlist[idx],
        #                                                                                  dtype=np.float64,
        #                                                                                  hwcorder=False, converter=None)
        #     startchannel += nchannel
        # if self.isTrain:
        #     label=gtReader.readPatchAsNumpy(initx,inity,self.cutoff_size,height=self.cutoff_size,bandlst=[],dtype=np.uint8,hwcorder=False,converter=None)
        #     assert img.shape[1:] == label.shape
        #     if self.div:
        #         label = np.array(label) / 255
        #     else:
        #         label = np.array(label)
        # else:
        #     label=None
        # if not self.DataAug is None and not self.DataAug is False:
        #     img,label = self.DataAug.aug(img,label)
        # if self.isTrain and self.DataAug:
        #     # #flip
        #     if self.isMirror:
        #         isHorizonFlip=self._randomBool()
        #         isVerticalFlip=self._randomBool()
        #         if isHorizonFlip:
        #             img=img[:,::-1,:]
        #             label=label[::-1,:]
        #         if isVerticalFlip:
        #             img = img[:,:,::-1]
        #             label=label[:,::-1]
        #     #rotate
        #     rotateAngle=self._randomRotate()
        #     if self.isRotate and not rotateAngle==0:
        #         for i in range(img.shape[0]):
        #             img[i,...] =np.rot90(img[i,...],rotateAngle)
        #         label=np.rot90(label,rotateAngle)
        img, label = super(DeepLabReader, self)._read_img(sample, initx, inity)
        label = label[::self.samplerate, ::self.samplerate]
        return (img,label)