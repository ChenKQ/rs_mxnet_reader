from osgeo import gdal
from osgeo.gdalconst import *
import numpy as np
import random
import sys,os
import cv2
from IImageReader import IImageReader



def genRandom(org_size,crop_size):
    return round(np.random.rand()*(org_size-crop_size))

def sampleOne(length):
    return random.sample(range(length),1)[0]

class labelViz(object):
    '''
    This class is used to convert formats between color maps which is used for visualization and label map which has only label like 0,1,2,3,....
    '''
    def __init__(self,tempfile):
        '''
        Initialization Method.
        :param tempfile: The template file.
        '''
        self.tempfile = tempfile
        with open(self.tempfile,'r') as f:
            lines=f.readlines()
        self.mapping=[]
        for line in lines:
            self.mapping.append(self.str2List(line))

    def str2List(self,s):
        '''
        This method converts from '255,255,0:2'->'255,255,0','2'->['255','255','0'],'2'->[255,255,0],1->{'color':[255,255,0], 'label':1}
        :param s: '255,255,0:2'
        :return: {'color':[255,255,0], 'label':1} . Color will be a list and label will be a number.
        '''
        color,lb=s.strip('\n').split(':')
        lb=int(lb)
        color=color.split(',')
        channels=[]
        for channel in color:
            channels.append(int(channel))
        mapping={}
        mapping['color']=channels
        mapping['label']=lb
        return mapping
    def unViz(self,colorMap,hwcorder=True):
        '''
        This method is used to convert from color map to label map
        :param colorMap: color map matrix(one channel used for two classes and three channels for multi-class are all right). Its shape should be (c,h,w)
        :return: label map (only one channel)
        '''
        if len(colorMap.shape) ==3:
            assert len(self.mapping[0]['color'])==3
            if hwcorder:
                colormaphwc = colorMap
            else:
                colormaphwc=np.swapaxes(colorMap,0,2)  ##c,h,w -->w,h,c
                colormaphwc=np.swapaxes(colormaphwc,0,1)  ##w,h,c -->h,w,c
            labelmap=np.zeros(colormaphwc.shape[0:2],dtype=np.uint8)
            for item in self.mapping:
                color=item['color']
                lb=item['label']
                idxtags = (colormaphwc==color)
                labelmap += (idxtags[...,0]*idxtags[...,1]*idxtags[...,2])*np.ones(idxtags.shape[0:2],dtype=np.uint8)*lb
            return labelmap
        else:
            assert len(self.mapping[0]['color'])==1
            labelmap = np.zeros(colorMap.shape, dtype=np.uint8)
            for item in self.mapping:
                color=item['color']
                lb=item['label']
                labelmap += (colorMap==color[0])*np.ones(colorMap.shape,dtype=np.uint8)*lb
            return labelmap
    def Viz(self,labelMap):
        '''
        This method is used to convert from label map to color map, it is used for visualization
        :param labelMap: label map matrix(one channel should be provided)
        :return: color map matrix which can be saved as three channel picture is returned. Its shape should be (h,w,c)
        '''
        colormap= np.zeros((labelMap.shape[0],labelMap.shape[1],3),dtype=np.uint8)
        for item in self.mapping:
            color=item['color']
            label=item['label']
            if len(color)==1:
                colormap[...,0] += (labelMap==label)*np.ones(colormap.shape[0:2],dtype=np.uint8)*color[0]
                colormap[...,1] += (labelMap==label)*np.ones(colormap.shape[0:2],dtype=np.uint8)*color[0]
                colormap[...,2] += (labelMap==label)*np.ones(colormap.shape[0:2],dtype=np.uint8)*color[0]
            else:
                assert len(color)==3
                colormap[...,0] += (labelMap==label)*np.ones(colormap.shape[0:2],dtype=np.uint8)*color[0]
                colormap[...,1] += (labelMap==label)*np.ones(colormap.shape[0:2],dtype=np.uint8)*color[1]
                colormap[...,2] += (labelMap==label)*np.ones(colormap.shape[0:2],dtype=np.uint8)*color[2]
        return colormap

    def convert2Lable(self,colormapfile,saveFolder,colorMap=True):
        '''
        This method is used to convert one color map file to label file, which is saved to saveFolder
        :param colormapfile: a file of color map
        :param saveFolder: destination save folder
        :return:
        '''
        dataset=gdal.Open(colormapfile,GA_ReadOnly)
        if dataset is None:
            print 'failed to open file: ',colormapfile
            return
        reader = GdalReader(colormapfile,colorMap=colorMap)
        colormap=reader.readImgAsNumpy()
        labelmap = self.unViz(colormap)
        basename=os.path.basename(colormapfile)
        cv2.imwrite(os.path.join(saveFolder,basename),labelmap)

    def convert2Color(self,labelfile,saveFolder):
        '''
        This method is used to convert one label map file to color map file, which is save to saveFolder.
        :param labelfile: a file of label map
        :param saveFolder:  destination save folder
        :return:
        '''
        gtset=gdal.Open(labelfile,GA_ReadOnly)
        if gtset is None:
            print 'failed to open file:',labelfile
            return
        reader=GdalReader(labelfile,colorMap=False)
        labelmap=reader.readImgAsNumpy()
        colormap=self.Viz(labelmap)
        basename=os.path.basename(labelfile)
        colorsave=np.zeros(colormap.shape,dtype=np.uint8)
        colorsave[...,0],colorsave[...,1],colorsave[...,2]=colormap[...,2],colormap[...,1],colormap[...,0]
        cv2.imwrite(os.path.join(saveFolder,basename),colorsave)

    def compare(self,map1,map2):
        img1=cv2.imread(map1,-1)
        img2=cv2.imread(map2,-1)
        if len(img1.shape) != len(img2.shape):
            print np.sum(img1-img2[...,0])
        else:
            print np.sum(img1-img2)

class GdalReader(IImageReader):
    def __init__(self,tiffile,rgbMean=[],colorMap=True):
        '''
        This is initialization method. It opens tiff file with gdal and get all bands. All thes bands are saved in self.bands.
        :param tiffile: imagery file to open
        :param rgbMean:
        :param colorMap: True of False. If color map is provided, it should be set True, else False.
        '''
        self.tiffile=tiffile
        self.rgbMean=rgbMean
        self.colorMap=colorMap
        self.bands=[]
        self.Open()
        self.getBands()

    def Open(self):
        '''
        This method opens tiffile with gdal and it is stored in self.dataset
        :return: None
        '''
        self.dataset = gdal.Open(self.tiffile, GA_ReadOnly)

    def getTransform(self):
        return self.dataset.GetGeoTransform()

    def getProjection(self):
        return self.dataset.GetProjection()

    def getBands(self):
        '''
        This method gets all bands of tiff file and these bands are stored in self.bands. It is lsit type.
        If tiff file is color map, there are three channels in self.bands, else there is only one channel and the length of self.bands should be one.
        :return: None
        '''
        assert not self.dataset is None
        for i in range(self.dataset.RasterCount):
            band=self.dataset.GetRasterBand(i+1)
            self.bands.append(band)
    def readPatchCHWOrder(self,startx,starty,width,height,dtype=np.uint8,bandlst=[],converter=None):
        '''
        This method returns the patch you want. If converter is not None, image will be converted as converter defined. Converter should be the type of labelViz
        :param startx: coordinate x of start point in origin image
        :param starty: coordinate y of start point in origin image
        :param width: the width to read
        :param height: the height to read
        :param bandlst: it figures out which bands need to be loaded and readed. If None or length is 0, all bands are needed. It starts from 1 instead of 0.
        :param converter: this converter should be type of labelViz, it is used to convert from color map to label map here
        :return: patch in h,w,c order
        '''
        if bandlst is None or len(bandlst)==0:
            if self.colorMap is False:
                assert len(self.bands)==1
                img = np.zeros((height,width),dtype=dtype)
                img[-min(0, starty):min(height, self.dataset.RasterYSize - starty),-min(0, startx):min(width, self.dataset.RasterXSize - startx)] \
                    = self.bands[0].ReadAsArray(max(0, startx), max(0, starty),min(width, width + startx, self.dataset.RasterXSize - startx),min(height, height + starty, self.dataset.RasterYSize - starty))
                if not converter is None:
                    assert len(converter.mapping[0]['color'])==1
                    img = converter.unViz(img)
            else:
                img = np.zeros((self.dataset.RasterCount,height, width),dtype=dtype)
                for idx,band in enumerate(self.bands):
                    img[idx,-min(0,starty):min(height,self.dataset.RasterYSize-starty),-min(0,startx):min(width,self.dataset.RasterXSize-startx)] \
                        = band.ReadAsArray(max(0,startx), max(0,starty), min(width,width+startx,self.dataset.RasterXSize-startx), min(height,height+starty,self.dataset.RasterYSize-starty))
                if not converter is None:
                    assert len(self.bands)==len(converter.mapping[0]['color'])
                    img = converter.unViz(img,False)
        else:
            img = np.zeros((len(bandlst), height, width), dtype=dtype)
                ## only bands in bandlst are get
            for idx, band in enumerate(bandlst):
                img[idx,-min(0, starty):min(height, self.dataset.RasterYSize - starty),-min(0, startx):min(width, self.dataset.RasterXSize - startx)] \
                    = self.bands[band].ReadAsArray(max(0, startx), max(0, starty),min(width, width + startx, self.dataset.RasterXSize - startx),min(height, height + starty, self.dataset.RasterYSize - starty))
            if not converter is None:
                assert len(converter.mapping[0]['color'])==len(bandlst)
                img = converter.unViz(img,False)
        if not self.rgbMean is None and len(self.rgbMean)>0:
            for idx,v in enumerate(self.rgbMean):
                img[idx,...] -=v
        return img
    def readPatchAsNumpy(self,startx,starty,width,height,bandlst=[],dtype=np.uint8,hwcorder=True,converter=None):
        '''
        This method returns the patch you want. If converter is not None, image will be converted as converter defined. Converter should be the type of labelViz
        :param startx: coordinate x of start point in origin image
        :param starty: coordinate y of start point in origin image
        :param width: the width to read
        :param height: the height to read
        :param bandlst: it figures out which bands need to be loaded and readed. If None or length is 0, all bands are needed. It starts from 1 instead of 0.
        :param hwcorder: True or False. If the returned img in h,w,c order
        :param converter: this converter should be type of labelViz, it is used to convert from color map to label map here
        :return: patch in h,w,c order
        '''
        if not hwcorder:
            return self.readPatchCHWOrder(startx,starty,width,height,dtype,bandlst,converter)
        if bandlst is None or len(bandlst)==0:
            if self.colorMap is False:   #grap image
                assert len(self.bands)==1
                img = np.zeros((height,width),dtype=dtype)
                img[-min(0, starty):min(height, self.dataset.RasterYSize - starty),-min(0, startx):min(width, self.dataset.RasterXSize - startx)] \
                    = self.bands[0].ReadAsArray(max(0, startx), max(0, starty),min(width, width + startx, self.dataset.RasterXSize - startx),min(height, height + starty, self.dataset.RasterYSize - starty))
                if not converter is None:
                    assert len(converter.mapping[0]['color'])==1
                    img = converter.unViz(img, hwcorder=False)
            else:  #color image
                img = np.zeros((height, width,self.dataset.RasterCount),dtype=dtype)
                for idx,band in enumerate(self.bands):
                    img[-min(0,starty):min(height,self.dataset.RasterYSize-starty),-min(0,startx):min(width,self.dataset.RasterXSize-startx),idx] \
                        = band.ReadAsArray(max(0,startx), max(0,starty), min(width,width+startx,self.dataset.RasterXSize-startx), min(height,height+starty,self.dataset.RasterYSize-starty))
                if not converter is None:
                    assert len(self.bands)==len(converter.mapping[0]['color'])
                    img = converter.unViz(img,hwcorder=False)
        else:
            img = np.zeros((height, width, len(bandlst)), dtype=dtype)
            ## only bands in bandlst are get
            for idx, band in enumerate(bandlst):
                img[-min(0, starty):min(height, self.dataset.RasterYSize - starty),-min(0, startx):min(width, self.dataset.RasterXSize - startx), idx] \
                    = self.bands[band].ReadAsArray(max(0, startx), max(0, starty),min(width, width + startx, self.dataset.RasterXSize - startx),min(height, height + starty, self.dataset.RasterYSize - starty))
            if not converter is None:
                assert len(converter.mapping[0]['color'])==len(bandlst)
                img = converter.unViz(img)
        if not self.rgbMean is None and len(self.rgbMean)>0:
            img -= self.rgbMean
        return img

    def readImgAsNumpy(self,bandlst=[],dtype=np.uint8,hwcorder=True,converter=None):
        '''
        Read the whole image in numpy format.
        :param bandlst: it figures out which bands need to be loaded and readed. If None or length is 0, all bands are needed. It starts from 1 instead of 0.
        :param hwcorder: True or False
        :param converter: this converter should be type of labelViz, it is used to convert from color map to label map here
        :return:
        '''
        return self.readPatchAsNumpy(0,0,self.dataset.RasterXSize,self.dataset.RasterYSize,bandlst,dtype,hwcorder,converter)

    def sumByChannel(self):
        img=self.readImgAsNumpy(bandlst=[],hwcorder=False,converter=None)
        sum = np.sum(img,axis=(1,2))
        return (sum,self.dataset.RasterXSize*self.dataset.RasterYSize)

    def getNChannel(self):
        return self.dataset.RasterCount

    def getSize(self):
        return (self.dataset.RasterYSize,self.dataset.RasterXSize)

    @classmethod
    def write(cls,outputPath,nbands,proj,trans,dataNumpy,gdalDType=gdal.GDT_Float32,npDType=np.float32):
        driver = gdal.GetDriverByName('GTiff')
        if driver is None:
            return
        if len(dataNumpy.shape) == 3:
            (height, width) = (dataNumpy.shape[1], dataNumpy.shape[2])
        else:
            (height, width) = dataNumpy.shape
        out_data_set = driver.Create(outputPath, width, height, nbands, gdalDType)
        out_data_set.SetProjection(proj)
        out_data_set.SetGeoTransform(trans)
        for idx in range(nbands):
            out_band = out_data_set.GetRasterBand(idx + 1)
            if nbands > 1:
                data = np.asarray(dataNumpy[idx, ...], dtype=npDType)
            else:
                data = np.asarray(dataNumpy[...], dtype=npDType)
            out_band.WriteArray(data)



def convertColor2Label():
    isprstemp='isprs.txt'
    converter=labelViz(isprstemp)
    watchdir='/home/yanml/mxtest/semantic/Data/vaihingen/gt'
    gt_label_dir='/home/yanml/mxtest/semantic/Data/vaihingen/gt_label'
    files=os.listdir(watchdir)
    for f in files:
        print f
        converter.convert2Lable(os.path.join(watchdir,f),gt_label_dir)

def convertLable2Color():
    isprstemp='isprs.txt'
    converter=labelViz(isprstemp)
    watchdir='/home/yanml/mxtest/semantic/Data/vaihingen/gt_label'
    gt_color_dir='/home/yanml/mxtest/semantic/Data/vaihingen/gt_color'
    files=os.listdir(watchdir)
    for f in files:
        print f
        converter.convert2Color(os.path.join(watchdir,f),gt_color_dir)

def convertColor2Label_building():
    isprstemp='building.txt'
    converter=labelViz(isprstemp)
    watchdir='/home/yanml/mxtest/semantic/Data/gt'
    gt_label_dir='/home/yanml/mxtest/semantic/Data/gt/gt_label'
    files=os.listdir(watchdir)
    for f in files:
        if f.endswith('.bmp'):
            print f
            converter.convert2Lable(os.path.join(watchdir,f),gt_label_dir,False)

def convertLable2Color_building():
    isprstemp='building.txt'
    converter=labelViz(isprstemp)
    watchdir='/home/yanml/mxtest/semantic/Data/gt/gt_label'
    gt_color_dir='/home/yanml/mxtest/semantic/Data/gt/gt_color'
    files=os.listdir(watchdir)
    for f in files:
        print f
        converter.convert2Color(os.path.join(watchdir,f),gt_color_dir)

def calMean(watchdirs,nchannel):
    sum = np.zeros((nchannel,))
    count = 0L
    for watchdir in watchdirs:
        files = os.listdir(watchdir)
        for f in files:
            if not f.endswith('.jpg'):
                continue
            reader=GdalReader(os.path.join(watchdir,f))
            (s,c) = reader.sumByChannel()
            sum +=s
            count += c
    sum /= count
    return sum


if __name__ =='__main__':
    '''
    sys.argv[1]: templatefile,it can be 'building.txt' or 'isprs.txt'
    sys.argv[2]: watchdir
    sys.argv[3]: save directory
    sys.argv[4]: (0 or 1, 1 for True) is this a three channel map?
    sys.argv[5]: (0 or 1, 1 for True) 1:convert2Label; 0: convert2Color
    '''
    reader = GdalReader('/mnt/sdb/chenkq/isprs/vaihingen/ndsm/dsm_09cm_matching_area13.tif',colorMap=False)
    img = reader.readImgAsNumpy(dtype=np.float64,hwcorder=True)
    print reader.dataset.bands[0]
    # isprstemp=sys.argv[1]
    # converter = labelViz(isprstemp)
    # watchdir = sys.argv[2]
    # savedir=sys.argv[3]
    # colorMap=sys.argv[4]
    # toLabel=sys.argv[5]
    # files=os.listdir(watchdir)
    # for f in files:
    #     if os.path.isfile(os.path.join(watchdir,f)):
    #         print f
    #         if int(toLabel)==1:
    #             print 'convert to Label Map'
    #             converter.convert2Lable(os.path.join(watchdir, f), savedir, bool(int(colorMap)))
    #         else:
    #             print 'convert to Color Map'
    #             converter.convert2Color(os.path.join(watchdir,f),savedir)

