from abc import ABCMeta,abstractmethod

class IImageReader(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def Open(self):
        pass

    @abstractmethod
    def getTransform(self):
        pass

    @abstractmethod
    def getProjection(self):
        pass

    @abstractmethod
    def readPatchAsNumpy(self,startx,starty,width,height,bandlst,dtype,hwcorder=True,converter=None):
        pass

    @abstractmethod
    def readImgAsNumpy(self,bandlst,dtype,hwcorder=True,converter=None):
        pass

    @abstractmethod
    def sumByChannel(self):
        pass

    @abstractmethod
    def getNChannel(self):
        pass

    @abstractmethod
    def getSize(self):
        pass