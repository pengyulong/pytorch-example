import torch
import torch.nn as nn

class DenseFeature(object):
    __slot__ = ()
    def __init__(self,name,dimension=1,dtype='float32'):
        self.name = name
        self.dimension = dimension
        self.dtype = dtype

    def get_name(self):
        return self.name

    def get_dtype(self):
        return self.dtype

    def get_dimension(self):
        return self.dimension


class SparseFeature(object):
    __slot__ = ()
    def __init__(self,name,vocab_size,embed_dim=4,dytpe='int32'):
        self.name = name
        self.vocab_size = vocab_size
        if self.embed_dim == 'auto':
            self.embed_dim = 6 * int(pow(self.vocab_size, 0.25))
        self.dtype = dtype

    def get_dim(self):
        return self.embed_dim

    def get_name(self):
        return self.get_name



