import mxnet as mx
# import RSReader
from SimpleBatch import SimpleBatch
import numpy as np
import logging
from GraphPatchIter import GraphPatchIter

class GraphPatchIter2(GraphPatchIter):
    def __init__(self, datareader, patchsize, dst_size, step, sample_idx=0):
        super(GraphPatchIter2,self).__init__(datareader, patchsize, dst_size, step, sample_idx)
        self.__datareader = datareader
        self.count = 0
        self.batchsize = datareader.batch_size
        self.allAreas = None
        self.readAreas = None
        self.labels = None
        self.imgs = None
        self.calAreas()

    def calAreas(self):
        self.allAreas = []
        self.readAreas = []
        while True:
            if super(GraphPatchIter2,self).next_idx():
                area = self.write_area()
                self.allAreas.append(area)
                readarea = self.calReadAreas()
                self.readAreas.append(readarea)
                self.count += 1
            else:
                self.reset()
                return

    def calReadAreas(self):
        if self.colidx!=self.xl:
            startx=self.step*self.colidx-self.pad
        else:
            startx=self.data_shape[1]+self.pad-self.patchsize
        if self.rowidx!=self.yl:
            starty=self.step*self.rowidx-self.pad
        else:
            starty=self.data_shape[0]+self.pad-self.patchsize
        return (starty,startx)

    def reset(self):
        super(GraphPatchIter2, self).reset()

    def get_img_label(self):
        imgs = []
        labels = []
        if self.cursor >= self.count - self.batchsize:
            self.cursor = self.count - self.batchsize - 1
        for idx in range(0,self.batchsize):
            self.cursor += 1
            (starty,startx) = self.readAreas[self.cursor]
            (img,label) = self.__datareader.read_img(self.sample,startx,starty)
            imgs.append(img)
            labels.append(label)
        self.imgs = np.asarray(imgs)  # img's shape should be (batch,channel,height,width)
        self.labels = np.asarray(labels,dtype=np.uint8)
        # print self.colidx,self.xl,starty,endy,startx,endx
    def get_img(self):
        return self.imgs

    def get_label(self):
        return self.labels

    def get_score(self):
        img=self.get_img()
        score=self.iter_predict(img)
        return score

    def next_idx(self):
        if self.cursor>=self.count-1:
            return False
        return True

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
        return output

    def next(self):
        if self.next_idx():
            self.get_img_label()
            coords = self.allAreas[self.cursor-self.batchsize+1:self.cursor+1]
            score = self.get_score()
            return (score,self.labels,self.imgs,coords)
        else:
            raise StopIteration()

