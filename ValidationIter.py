import mxnet as mx
from GraphPatchIter2 import GraphPatchIter2
from SimpleBatch import SimpleBatch


class ValidationIter(object):
    def __init__(self,datareader,patchsize,dst_size,step,sample_length):
        self.__cursor = 0
        self.__datareader = datareader
        self.__patchsize = patchsize
        self.__dst_size = dst_size
        self.__step = step
        self.__patchiter = GraphPatchIter2(self.__datareader,self.__patchsize,self.__dst_size,self.__step,sample_idx=self.__cursor)
        self.__sample_length = sample_length

    def __iter__(self):
        return self

    def reset(self):
        self.__cursor = 0
        self.__patchiter = GraphPatchIter2(self.__datareader, self.__patchsize, self.__dst_size, self.__step,
                                           sample_idx=self.__cursor)

    def next_idx(self):
        if self.__cursor >= self.__sample_length:
            return False
        else:
            return True

    def next(self):
        if self.__patchiter.next_idx():
            self.__patchiter.get_img_label()
            img = self.__patchiter.get_img()
            label = self.__patchiter.get_label()
            return SimpleBatch([mx.nd.array(img)], [mx.nd.array(label)])
            # return SimpleBatch([mx.nd.array(self.__patchiter.get_img())], [mx.nd.array(self.__patchiter.get_label())])
        else:
            self.__cursor +=1
            if self.next_idx():
                self.__patchiter = GraphPatchIter2(self.__datareader, self.__patchsize, self.__dst_size, self.__step,
                                                   sample_idx=self.__cursor)
                self.__patchiter.get_img_label()
                return SimpleBatch([mx.nd.array(self.__patchiter.get_img())], [mx.nd.array(self.__patchiter.get_label())])
            else:
                raise StopIteration