#coding:utf-8 ��������ע��
# Author: yongyuan.name
import numpy as np
import cv2
from numpy import linalg as LA
import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

#��Բ�ͬ���ࣺ
clas='xuebi'
#query_num=60 #queryͼƬ������
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

input_shape = [224, 224, 3]
model = VGG16(weights = 'imagenet', input_shape = (input_shape[0], input_shape[1], input_shape[2]), pooling = 'max', include_top = False)

def extract_feat(img, model):
    #pdb.set_trace()
    input_shape = (224, 224, 3)
    #img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    #pdb.set_trace()

    img=cv2.resize(img,(input_shape[0], input_shape[1]),interpolation=cv2.INTER_CUBIC)
    img = keras.preprocessing.image.img_to_array(img)
    #img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feat = model.predict(img)
    norm_feat = feat[0]/LA.norm(feat[0])#Ҳ������ǹ�һ����
    return norm_feat

a = np.loadtxt('/mnt/disk1/gyl/new_retail_0226/goods16_data_2018_02_24_1603.xml')
#a = np.loadtxt('/mnt/disk1/gyl/caffe/result.xml')
feats = a.reshape(215,512)

touch_num=0 #top1���е�ͼƬ������

for i in range(1,query_num+1):
    name2 = '0000'
    name3 = name2 + str(i)
    name3 = name3[-4:]
    query_image = cv2.imread('/mnt/disk1/gyl/image_test/product_retrieval/groundtruth_roi0308/'+clas+'/'+name3+'.jpg')
    queryVec = extract_feat(query_image,model)
    scores = np.dot(queryVec, feats.T)
    scores = np.array(scores)
    rank_ID = np.argsort(-scores)#�Ӵ�С��������  rank_ID[0]��ľ��ǵ÷����ͼƬ�����
    rank_score = scores[rank_ID]#�Ѿ������ƶȰ��Ӵ�С��˳�������ˣ�rank_score[0]������ߵĵ÷֣��������Ƶ�ͼƬ
    for k in range(range1,range2): #groundtruth�����
        if rank_ID[0]+1==k:
            touch_num=touch_num+1
            print(rank_ID[0]+1,rank_score[0])#rank_ID[0]��0��ʼ�����ҵ�ͼƬ��Ŵ�1��ʼ�ģ���+1
precision=float(touch_num)/query_num
print("The precision is:",'%.4f'%precision)


