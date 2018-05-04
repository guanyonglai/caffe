# -*- coding: utf-8 -*-
# Author: yongyuan.name
import numpy as np
from numpy import linalg as LA
import caffe

caffe.set_mode_gpu()
output = []
res_feats = []

#针对不同的类：
clas='nongfushanquan'
#query_num=60 #query图片的张数
if clas=='nongfushanquan':
    query_num=60
    range1,range2=89,98
elif clas=='laotansuancai':
    query_num=98
    range1,range2=78,89
elif clas=='kele':
    query_num=46
    range1,range2=18,29
elif clas=='pijiu':
    query_num=132
    range1,range2=98,127
elif clas=='xuebi':
    query_num=168
    range1,range2=201,216


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

def Feat_Extract(image,net):
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    output = np.squeeze(output['pool5/7x7_s1'])
    #output = Norm(output)
    output = output/LA.norm(output)
    return output

def Norm(a):
    a = np.array(a)
    amin, amax = a.min(), a.max()
    a = (a-amin)/(amax-amin)
    return a

a = np.loadtxt('/mnt/disk1/gyl/caffe/result.xml')
feats = a.reshape(215,1024)

touch_num=0 #top1命中的图片的张数

for i in range(1,query_num+1):
    name2 = '0000'
    name3 = name2 + str(i)
    name3 = name3[-4:]
    queryImg = caffe.io.load_image('/mnt/disk1/gyl/image_test/product_retrieval/groundtruth_roi0308/'+clas+'/'+name3+'.jpg')
    # extract query image's feature, compute simlarity score and sort
    queryVec = Feat_Extract(queryImg,net)
    scores = np.dot(queryVec, feats.T)
    scores = np.array(scores)
    rank_ID = np.argsort(-scores)#从大到小逆序排序  rank_ID[0]存的就是得分最高图片的序号
    rank_score = scores[rank_ID]#已经把相似度按从大到小的顺序排列了，rank_score[0]就是最高的得分，即最相似的图片
    for k in range(range1,range2): #groundtruth的序号
        if rank_ID[0]+1==k:
            touch_num=touch_num+1
            print(rank_ID[0]+1,rank_score[0])#rank_ID[0]从0开始，而我的图片编号从1开始的，故+1
precision=float(touch_num)/query_num
print("The precision is:",'%.4f'%precision)


