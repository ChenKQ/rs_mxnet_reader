import sys
sys.path.insert(0,'../..')
import logging
import mxnet as mx
from SymbolNew.segmentation.symbol_resscnn import get_symbol
from DataReader.DUCReader import DUCReader
from DataReader.ValidationIter import ValidationIter
from Measure.segLoss import MuLableCrossEntropy
from myoperator.poly_scheduler import PolyScheduler

def train_finetune(net,load_net_prefix,save_net_prefix,trainlst_name,vallst_name,batchsize,nepoch,trainepochiter,sample_length,data_root,rgb_mean,cropsize,begin_epoch=0,
                   ctx=None,opt=None,lr_scheduler=None,data_aug=None,
                   **kwargs):
    # prepare logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #prepare data and device
    train_iter= DUCReader(trainlst_name,batchsize,epochiter=trainepochiter,div=False,data_root=data_root,withGdal=True,samplerate=8,cell=2,
                 rgbMeans=rgb_mean,cutoff_size=cropsize,isTrain=True,DataAug=data_aug)
    val_iter=DUCReader(vallst_name,batchsize*2,div=False,data_root=data_root,withGdal=True,samplerate=8,cell=2,
                 rgbMeans=rgb_mean,cutoff_size=cropsize,isTrain=True,DataAug=data_aug)
    validation_iter = ValidationIter(val_iter,patchsize=cropsize,dst_size=cropsize,step=cropsize,sample_length=sample_length)
    if ctx==None:
        ctx=[mx.gpu(2)]
    else:
        ctx=ctx

    #metrics
    ce = MuLableCrossEntropy(class_weight=[])
    acc=mx.metric.Accuracy()
    multimetric=mx.metric.CompositeEvalMetric()
    multimetric.add(ce)
    multimetric.add(acc)

    #set the training process
    # netprefix=net_save_prefix
    checkpoint=mx.callback.do_checkpoint(save_net_prefix,period=10)
    batchend=mx.callback.Speedometer(batchsize,10)

    #check optimizer
    if opt is None:
        logging.info('opt should be set')
        return
    if lr_scheduler is None:
        logging.info('learning scheduler should be set')
        return

    arg_params = None
    aux_params = None
    if not net is None and load_net_prefix is None:  #1,0
        mod=mx.module.Module(symbol=net,data_names=['data'],label_names=['softmax_label'],context=ctx,logger=logger)
    elif not net is None and not load_net_prefix is None: #1,1
        _, arg_params, aux_params = mx.model.load_checkpoint(load_net_prefix, begin_epoch)
        mod = mx.module.Module(symbol=net, data_names=['data'], label_names=['softmax_label'], context=ctx,
                               logger=logger)
    else:
        mod = mx.module.Module.load(load_net_prefix, begin_epoch, context=ctx, logger=logger)
    mod.fit(train_data=train_iter,eval_data=validation_iter,eval_metric=multimetric,
            optimizer=opt, optimizer_params=lr_scheduler,arg_params=arg_params,aux_params=aux_params,
            epoch_end_callback=checkpoint,batch_end_callback=batchend,
            begin_epoch=begin_epoch,num_epoch=nepoch,**kwargs)



if __name__ == '__main__':
    net = get_symbol(num_classes=6, num_layers=34, strides=[1,2,1,1],atrous=[1,1,2,2],reshaped=True,scnn_compress=None,samplerate=8,cell=2,conv_workspace=256)
    batchsize= 16
    ctx=[mx.gpu(12),mx.gpu(13)]
    init = mx.initializer.MSRAPrelu('avg')
    trainlst_name='/mnt/chenkq/data/potsdam/trainset.lst'
    vallst_name='/mnt/chenkq/data/potsdam/valset.lst'
    data_root = '/run/chenkq/data/potsdam'
    net_prefix = '/mnt/chenkq/models/potsdam/potsdam_resscnn34'
    rgb_mean = [[37.257698], [86.55175022, 92.5452277, 85.91596489, 97.63989512],[0]]
    opt = 'sgd'
    lr_sch =(('learning_rate',0.01),('momentum',0.9),('wd',0.00005),('lr_scheduler',PolyScheduler(max_iter=150000,power=0.9,stop_factor_lr=1e-8)))

    train_finetune(net=net,load_net_prefix=None,save_net_prefix=net_prefix,
                   trainlst_name=trainlst_name,vallst_name=vallst_name,batchsize=batchsize,
                   nepoch=300,trainepochiter=500,sample_length=7,data_root=data_root,rgb_mean=rgb_mean,cropsize=224,begin_epoch=0,
                   ctx=ctx,opt=opt,lr_scheduler=lr_sch,data_aug=None,
                   initializer=init,allow_missing=True,force_init=True)