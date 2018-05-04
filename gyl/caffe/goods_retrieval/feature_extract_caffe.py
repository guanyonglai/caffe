# -*- coding: UTF-8 -*-
import caffe                                                      #导入caffe包
import numpy as np
from numpy import linalg as LA
#import matplotlib.pyplot as plt

caffe.set_mode_gpu()
output = []
res_feats = []

model_def = '/mnt/disk1/gyl/caffe/yi+shopping.prototxt'
model_weights = '/mnt/disk1/gyl/caffe/yi+shopping.caffemodel'
mu = np.load('./ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  #对所有像素值取平均以此获取BGR的均值像素值
print 'mean-subtracted values:', zip('BGR', mu)
net = caffe.Net(model_def,      # defines the structure of the model
            model_weights,  # contains the trained weights
            caffe.TEST)     # use test mode (e.g., don't perform dropout)
transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)            #对于每个通道，都减去BGR的均值像素值
transformer.set_raw_scale('data', 255)      #将像素值从[0,255]变换到[0,1]之间
transformer.set_channel_swap('data', (2,1,0))  #交换通道，从RGB变换到BGR

def Norm(a):
    a = np.array(a)
    amin, amax = a.min(), a.max()
    a = (a-amin)/(amax-amin)
    return a

def Feat_Extract(image,net):
    #image=caffe.io.resize(224,224)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    output = np.squeeze(output['pool5/7x7_s1'])
    #output = Norm(output)
    output = output/LA.norm(output)
    return output

for i in range(1,216):
    name1 = 'goods_retrieval_'
    name2 = '0000'
    name3 = name2 + str(i)
    name3 = name3[-4:]
    name = name1 + name3
    image = caffe.io.load_image('/mnt/disk1/gyl/image_test/product_retrieval/Goods_db/'+name+'.jpg')
    output = Feat_Extract(image,net)
    res_feats.append(output)

#res_feats = np.transpose(res_feats)
np.savetxt("result.xml", res_feats)

#print(output.shape)
print(output)
