import cv2
import os,sys

from IImageReader import IImageReader
import numpy as np

class OpenCVReader(IImageReader):
    def __init__(self,imgfile,rgbMean=[],colorMap=True):
        self.imgfile = imgfile
        self.rgbMean=rgbMean
        self.colorMap = colorMap

    def Open(self):
        pass

    def getTransform(self):
        pass

    def getProjection(self):
        pass

    def _readPatchCHWOrder(self, startx, starty, width, height, dtype, bandlst=[], converter=None):
        img = cv2.imread(self.imgfile, -1)  # (height,width,channel)
        if width<0:
            width = img.shape[1]
        if height<0:
            height = img.shape[0]
        if bandlst is None or len(bandlst)==0:
            if self.colorMap is False:
                assert len(img.shape)==2
                ret = np.zeros((height,width),dtype=dtype)
                ret[-min(0, starty):min(height, img.shape[0] - starty),-min(0, startx):min(width, img.shape[1] - startx)] \
                    = img[max(0,starty):max(0,starty)+min(height, height + starty, img.shape[0] - starty),max(0, startx):max(0, startx)+min(width, width + startx, img.shape[1] - startx)]
                if not converter is None:
                    assert len(converter.mapping[0]['color'])==1
                    ret = converter.unViz(ret)
            else:
                assert len(img.shape) == 3
                ret = np.zeros((img.shape[3],height, width),dtype=dtype)
                for idx in range(img.shape[2]):
                    ret[idx,-min(0,starty):min(height,img.shape[0]-starty),-min(0,startx):min(width,img.shape[1]-startx)] \
                        = img[max(0,starty):max(0,starty)+min(height, height + starty, img.shape[0] - starty),max(0, startx):max(0, startx)+min(width, width + startx, img.shape[1] - startx),idx]
                if not converter is None:
                    assert len(img.shape[2])==len(converter.mapping[0]['color'])
                    ret = converter.unViz(ret,False)
        else:
            ret = np.zeros((len(bandlst), height, width), dtype=dtype)
                ## only bands in bandlst are get
            for idx, band in enumerate(bandlst):
                ret[idx,-min(0, starty):min(height, img.shape[0] - starty),-min(0, startx):min(width, img.shape[1] - startx)] \
                    = img[max(0, starty):max(0, starty) + min(height, height + starty, img.shape[0] - starty),max(0, startx):max(0, startx) + min(width, width + startx, img.shape[1] - startx), idx]
            if not converter is None:
                assert len(converter.mapping[0]['color'])==len(bandlst)
                ret = converter.unViz(ret,False)
        if not self.rgbMean is None and len(self.rgbMean)>0:
            for idx,v in enumerate(self.rgbMean):
                ret[idx,...] -=v
        return ret

    def readPatchAsNumpy(self,startx,starty,width,height,bandlst,dtype,hwcorder=True,converter=None):
        if not hwcorder:
            return self.readPatchCHWOrder(startx,starty,width,height,dtype,bandlst,converter)
        img = cv2.imread(self.imgfile,-1)
        if width<0:
            width = img.shape[1]
        if height<0:
            height = img.shape[0]
        if bandlst is None or len(bandlst)==0:
            if not self.colorMap:
                assert len(img.shape) == 2
                ret = np.zeros((height, width), dtype=dtype)
            else:  #color image
                assert len(img.shape)==3
                ret = np.zeros((height, width,img.shape[2]),dtype=dtype)
            ret[-min(0, starty):min(height, img.shape[0] - starty), -min(0, startx):min(width, img.shape[1] - startx)] \
                = img[max(0, starty):max(0, starty) + min(height, height + starty, img.shape[0] - starty),
                  max(0, startx):max(0, startx) + min(width, width + startx, img.shape[1] - startx)]
            if not converter is None:
                assert len(converter.mapping[0]['color']) == 1
                ret = converter.unViz(ret)
        else:
            ret = np.zeros((height, width, len(bandlst)), dtype=dtype)
            ## only bands in bandlst are get
            for idx, band in enumerate(bandlst):
                ret[-min(0, starty):min(height, img.shape[0] - starty),-min(0, startx):min(width, img.shape[1] - startx), idx] \
                    = img[max(0, starty):max(0, starty) + min(height, height + starty, img.shape[0] - starty),
                  max(0, startx):max(0, startx) + min(width, width + startx, img.shape[1] - startx),band]
            if not converter is None:
                assert len(converter.mapping[0]['color'])==len(bandlst)
                ret = converter.unViz(ret)
        if not self.rgbMean is None and len(self.rgbMean)>0:
            ret -= self.rgbMean
        return ret

    def readImgAsNumpy(self,bandlst,dtype,hwcorder=True,converter=None):
        return self.readPatchAsNumpy(0, 0, -1, -1, bandlst, dtype, hwcorder,
                                     converter)

    def sumByChannel(self):
        img=self.readImgAsNumpy(bandlst=[],hwcorder=False,converter=None)
        sum = np.sum(img,axis=(1,2))
        return (sum,img.shape[0]*img.shape[1])

    def getNChannel(self):
        img = cv2.imread(self.imgfile, -1)
        if len(img.shape)==2:
            return 1
        else:
            return img.shape[2]
    def getSize(self):
        img = cv2.imread(self.imgfile, -1)
        return img.shape[0:2]


if __name__ =='__main__':
    imgfile = '/mnt/chenkq/data/GSet/trainset/10_top.tif'
    reader = OpenCVReader(imgfile)
    img = reader.readImgAsNumpy(bandlst=[], dtype=np.uint8, hwcorder=True, converter=None)