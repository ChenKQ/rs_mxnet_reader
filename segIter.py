from mxnet.io import DataIter
import numpy as np
import cv2

class SegIter(DataIter):
    '''
    The second edition of data reader. It is derived from DataLoader.FileIter. It has been deprecated now.
    It expands its batch size to any.
    '''
    def __init__(self,root_dir,flist_name,batch_size,div=True,
                 rgb_mean=(105.229,105.247,97.835),cutoff_size=None,
                 data_name='data',
                 label_name='softmax_label'):
        super(DataIter,self).__init__()
        self.root_dir=root_dir
        self.flist_name=flist_name
        self.batch_size=batch_size
        self.div=div
        self.mean=np.array(rgb_mean)
        self.cutoff_size=cutoff_size
        self.data_name=data_name
        self.label_name=label_name
        self.num_data=len(open(self.flist_name,'r').readlines())
        self.f= open(self.flist_name,'r')
        self.data,self.label=self._read()
        self.cursor=-1


    def _read(self):
        data={}
        label={}
        data[self.data_name]=[]
        label[self.label_name]=[]
        for i in range(self.batch_size):
            data_img_name,label_img_name=self.f.readline().strip('\n').split('\t')
            data_,label_=self._read_img(data_img_name,label_img_name)
            data[self.data_name].append(data_)
            label[self.label_name].append(label_)
        data[self.data_name]=np.asarray(data[self.data_name])
        label[self.label_name]=np.asarray(label[self.label_name])
        return data.items(),label.items()

    def _read_img(self,img_name,label_name):
        # img_pil=Image.open(img_name)
        # label_pil=Image.open(label_name)
        img_pil=cv2.imread(img_name,-1)
        label_pil=cv2.imread(label_name,-1)
        # assert img_pil.size==label_pil.size
        assert img_pil.shape[0:2]==label_pil.shape
        img_ndarray=np.array(img_pil,dtype=np.float32)
        if self.div:
            label_ndarray=np.array(label_pil)/255
        else:
            label_ndarray=np.array(label_pil)
        if self.cutoff_size is not None:
            pass
        reshaped_mean = self.mean.reshape(1,1,3)
        img_ndarray -= reshaped_mean   #(h,w,c)
        img_ndarray = np.swapaxes(img_ndarray,0,2)  #(c,w,h)
        img_ndarray = np.swapaxes(img_ndarray,1,2)  #(c,h,w)
        #img_ndarray = np.expand_dims(img_ndarray,axis=0)   #(1,c,h,w)
        #label_ndarray = np.expand_dims(label_ndarray,axis=0)  #(1,h,w)
        return (img_ndarray,label_ndarray)

    @property
    def provide_data(self):
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
        if self.cursor<=self.num_data-self.batch_size:
            return True
        else:
            return False

    def next(self):
        if self.iter_next():
            self.data,self.label=self._read()
            return {self.data_name: self.data[0][1],
                    self.label_name:self.label[0][1]}
        else:
            raise StopIteration
