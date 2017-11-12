import mxnet as mx
# import RSReader
from SimpleBatch import SimpleBatch
import numpy as np
import logging

class GraphPatchIter(object):
    '''
    This class is used to predict the whole image.
    '''
    def __init__(self,datareader,patchsize,dst_size,step,sample_idx=0):
        '''
        Initialization.
        :param predictor: Its type should be Inference.
        :param flist_name: Only one example in flist_name is supported by now.
        :param patchsize:
        :param step:
        :param rgb_mean:
        :param withAnswer:
        :param batchsize:
        '''
        self.__datareader=datareader
        self.sample_idx=sample_idx
        self.sample = self.__datareader.sampleStore.getOneSample(self.sample_idx)
        # self.sample = self.datareader.sampleStore.samples[self.sample_idx]
        self.data_shape=(self.sample['top'][0].dataset.RasterYSize,self.sample['top'][0].dataset.RasterXSize)
        self.crop_shape = (patchsize,patchsize)
        # self.score=np.zeros(self.crop_shape,dtype=np.float64)
        self.patchsize=patchsize
        self.dst_size=dst_size
        self.pad=(self.patchsize-self.dst_size)/2
        self.step=step
        self.xl=(self.data_shape[1]-self.patchsize+2*self.pad)/self.step+1
        self.yl=(self.data_shape[0]-self.patchsize+2*self.pad)/self.step+1
        self.overlap=max(self.dst_size-self.step,0)
        self.rowidx=0
        self.colidx=-1
        self.cursor=-1

    def __iter__(self):
        return self

    def bind(self,ctx,net_prefix,epoch):
        self.ctx=ctx
        # self.net=net
        # self.input_shapes = input_shapes
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.info("testing ")
        self.mod=mx.module.Module.load(prefix=net_prefix,epoch=epoch,context=self.ctx,logger=self.logger)
        # self.mod =mx.module.Module(symbol=net,data_names=data_name,label_names=label_name,context=ctx,logger=self.logger)
        # print self.datareader.provide_data, self.datareader.provide_label
        self.mod.bind(data_shapes=self.__datareader.provide_data, label_shapes=self.__datareader.provide_label,for_training=False)  # data_shapes=[('data',(batch,channel,height,width))]
        print 'Bind Done'

    def iter_predict(self,data_array):
        '''
        iter one step
        :param data_array: Its shape should be [batch,channel,height,width]. It should be the type of numpy.ndarray
        :return:
        '''
        data_array = mx.nd.array(data_array,ctx=self.ctx)
        data_batch = SimpleBatch([data_array],None)
        # data_batch.provide_data = self.datareader.provide_data
        # data_batch.provide_label =self.datareader.provide_label
        self.mod.forward(data_batch, is_train=False)
        output = self.mod.get_outputs()[0].asnumpy()
        return output[0]


    def next_idx(self):
        if not (self.colidx==self.xl and self.rowidx==self.yl):
            # self.cursor +=1
            if self.colidx==self.xl:
                self.colidx=0
                self.rowidx +=1
            else:
                self.colidx+=1
            return True
        else:
            return False
    def reset(self):
        # self.score=np.zeros(self.crop_shape,dtype=np.float64)
        self.rowidx=0
        self.colidx=-1
        self.cursor=-1

    def next(self):
        if self.next_idx():
            coords=self.write_area()
            score=self.get_score()
            (img,label) = self.get_img_label()
            return (score,label,img,coords)
        else:
            raise StopIteration()

    def get_img_label(self):
        if self.colidx!=self.xl:
            startx=self.step*self.colidx-self.pad
        else:
            startx=self.data_shape[1]+self.pad-self.patchsize
        if self.rowidx!=self.yl:
            starty=self.step*self.rowidx-self.pad
        else:
            starty=self.data_shape[0]+self.pad-self.patchsize
        # print startx,starty
        (img,label) = self.__datareader.read_img(self.sample,startx,starty)
        img = np.expand_dims(img,0)
        img = np.asarray(img)  # img's shape should be (batch,channel,height,width)
        # print self.colidx,self.xl,starty,endy,startx,endx
        return (img,label)

    def get_img(self):
        return self.get_img_label()[0]

    def get_label(self):
        return self.get_img_label()[1]

    def get_score(self):
        img=self.get_img()
        score=self.iter_predict(img)
        return score

    def write_area(self):
        if self.colidx==0:
            startx=0
            endx=self.dst_size-self.overlap/2
            fromstartx=0
            fromendx=self.dst_size-self.overlap/2
        elif self.colidx==self.xl:
            localoverlap=(self.step*(self.colidx-1)+self.dst_size)-(self.data_shape[1]-self.dst_size)
            # print localoverlap,
            startx=self.data_shape[1]-self.dst_size+localoverlap/2
            endx=self.data_shape[1]
            fromstartx=localoverlap/2
            fromendx=self.dst_size
        else:
            startx=self.step*self.colidx+self.overlap/2
            endx=self.step*self.colidx+self.dst_size-self.overlap/2
            fromstartx=self.overlap/2
            fromendx=self.dst_size-self.overlap/2

        if self.rowidx==0:
            starty=0
            endy=self.dst_size-self.overlap/2
            fromstarty=0
            fromendy=self.dst_size-self.overlap/2
        elif self.rowidx==self.yl:
            localoverlap=(self.step*(self.rowidx-1)+self.dst_size)-(self.data_shape[0]-self.dst_size)
            starty=self.data_shape[0]-self.dst_size+localoverlap/2
            endy=self.data_shape[0]
            fromstarty=localoverlap/2
            fromendy=self.dst_size
        else:
            starty=self.step*self.rowidx+self.overlap/2
            endy=self.step*self.rowidx+self.dst_size-self.overlap/2
            fromstarty=self.overlap/2
            fromendy=self.dst_size-self.overlap/2
        return [(starty,endy,startx,endx),(fromstarty,fromendy,fromstartx,fromendx)]