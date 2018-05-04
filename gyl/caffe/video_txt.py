
# -*- coding: utf-8 -*-
# Author: yongyuan.name
import numpy as np
from numpy import linalg as LA
import caffe

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

#modify here to fenerate perdict txt
query_file='phone_cut_test'

name_txt=open('/mnt/disk1/gyl/image_test/video_retrieval0426/query/'+ query_file  +'.txt','r')
names=name_txt.readlines()

name_ku_txt=open('/mnt/disk1/gyl/image_test/video_retrieval0426/txt_db/frame_name.txt','r')
names_ku=name_ku_txt.readlines()

a = np.loadtxt('/mnt/disk1/gyl/caffe/video_feat0426.xml')
feats = a.reshape(1341,1024)


for i in range(0,len(names)):
    query_name=names[i].strip('\r\n')
    print('query_name:',query_name)

    out_file=open('/mnt/disk1/gyl/image_test/video_retrieval0426/predict/'+query_file+'/'+query_name+'.txt','w')

    queryImg = caffe.io.load_image('/mnt/disk1/gyl/image_test/video_retrieval0426/query/'+query_file+'/'+query_name+'.jpg')
    print('shape:',queryImg.shape)
    # extract query image's feature, compute simlarity score and sort
    queryVec = Feat_Extract(queryImg,net)
    scores = np.dot(queryVec, feats.T)
    scores = np.array(scores)
    rank_ID = np.argsort(-scores)#从大到小逆序排序  rank_ID[0]存的就是得分最高图片的序号
    rank_score = scores[rank_ID]#已经把相似度按从大到小的顺序排列了，rank_score[0]就是最高的得分，即最相似的图片
    for k in range(0,8):
        print(rank_ID[k]+1,names_ku[rank_ID[k]].strip('\r\n'),rank_score[k])#rank_ID[0]从0开始，而我的图片编号从1开始的，故+1
        out_file.write(str(names_ku[rank_ID[k]].strip('\r\n')) + ' ' + str(rank_score[k])[:6] + '\n')
out_file.close()


