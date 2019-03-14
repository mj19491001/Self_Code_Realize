#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 22:44:02 2018

@author: jq_tongdun
"""

######################################神经网络###################################
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer #乳腺癌数据加载
from sklearn.cross_validation import train_test_split

#单隐藏层神经网络反向传播算法BP实践
#面向对象的思想，定义模型类的对应的方法
#基函数采用sigmoid函数
#求解方法基于梯度下降

class JQ_nnet():
    import numpy as np
    def sigmoid(self,x): #激活函数（sigmoid函数）
        return (1/(1+np.exp(-x)))
    def sigmoid_der(self,x): #sigmoid函数的导数
        return self.sigmoid(x)*(1-self.sigmoid(x))    
    def softmax(self,x): #多分类输出函数（softmax函数）
        return np.apply_along_axis(lambda x: np.exp(x)*1.0/np.exp(x).sum(),1,x)
    #def softmax_der(self,x): #softmax的导数（用不着而且不正确）
    #    return self.softmax(x)*(1-self.softmax(x))    
    def fit(self,x,y,
            hidden = 5, #隐藏层数量
            iter=5000, #迭代次数
            s=0.01, #步长
            decay = 0.001, #步长衰减因子
            batch_size = 100, #每次梯度下降抽取的样本量
            loss='binary_crossentropy', #分类损失函数（可选二分类binary_crossentropy和多分类crossentropy）
            error_iter=1000, #每迭代多少次打印当前loss值
            seed=10): #随机种子
        x = np.matrix(np.hstack((np.ones((x.shape[0],1)),x)))
        try:
            y.shape[1]
            y = np.matrix(y)
        except:
            y = np.matrix(y).T
        length1 = x.shape[1]        
        width1 = hidden
        length2 = hidden + 1
        width2 = y.shape[1]
        s_start = s
        np.random.seed(seed)
        K = np.random.rand(length1,width1)*2-1 #初始化参数：从输入层到隐含层
        np.random.seed(seed+100) #随机种子固定
        W = np.random.rand(length2,width2)*2-1 #初始化参数：从隐含层到输出层
        i = 1
        while i <= iter: #迭代过程
            np.random.seed(seed+i*10) #随机种子固定
            sample_idx = np.random.choice(x.shape[0],batch_size,replace=False) #随机梯度下降采样
            x_s = x[sample_idx,:] #输入层采样
            y_s = y[sample_idx,:] #标签采样
            if i % error_iter != 0:
                Z1_s = x_s.dot(K) #隐藏层计算（输入层*权值）
                H_s = np.hstack((np.ones((x_s.shape[0],1)),self.sigmoid(Z1_s))) #隐藏层计算（激活函数）
                Z2_s = H_s.dot(W) #输出层计算（隐藏层*权值）
                if loss == 'binary_crossentropy': #二分类问题流程
                    y_pre_s = self.sigmoid(Z2_s) #输出层计算（激活函数：sigmoid）
                else:
                    y_pre_s = self.softmax(Z2_s) #输出层计算（激活函数：softmax）
            else:
                Z1 = x.dot(K) #隐藏层计算（输入层*权值）
                H = np.hstack((np.ones((x.shape[0],1)),self.sigmoid(Z1))) #隐藏层计算（激活函数）
                Z2 = H.dot(W) #输出层计算（隐藏层*权值）
                H_s = H[sample_idx,:] #隐藏层采样
                x_s = x[sample_idx,:] #输入层采样
                Z1_s = Z1[sample_idx,:] #隐藏层采样
                if loss == 'binary_crossentropy': #二分类问题流程
                    y_pre = self.sigmoid(Z2) 
                    error = -(np.array(y) * np.log(np.array(y_pre))+(1-np.array(y))*np.log(1-np.array(y_pre))).sum()
                    print 'n_iter: ', i, ' binary_crossentropy: ', error
                else:
                    y_pre = self.softmax(Z2) 
                    error = -(np.array(np.log(y_pre))*np.array(y)).sum()
                    print 'n_iter: ', i, ' crossentropy: ', error
                y_pre_s = y_pre[sample_idx,:] #输出层采样
            der_Z1 = np.matrix(self.sigmoid_der(np.array(Z1_s)))
            gra_W = H_s.T.dot(np.matrix(np.array(y_pre_s-y_s)))/float(batch_size)#权值梯度计算（隐藏层——输出层）
            gra_K = x_s.T.dot(np.array(np.matrix(y_pre_s-y_s).dot(W[1:,:].T)) * np.array(der_Z1))/float(batch_size) #权值梯度计算（输入层——隐藏层）
            W = W - gra_W * s #权值更新（隐藏层——输出层）
            K = K - gra_K * s #权值更新（输入层——隐藏层）
            s = s_start / (1+i*decay) #步长衰减
            i += 1
        self.W = W
        self.K = K 
    def predict_prob(self,x): #预测函数（概率预测）
        x = np.matrix(np.hstack((np.ones((x.shape[0],1)),x)))
        Z1 = x.dot(self.K)
        H = np.hstack((np.ones((x.shape[0],1)),self.sigmoid(Z1)))
        Z2 = H.dot(self.W)
        if self.W.shape[1]>1:
            res = np.array(self.softmax(Z2))
        else:
            res = np.array(self.sigmoid(Z2)).T[0]
        return res
    def predict_type(self,x): #预测函数（类别预测）
        res_prob = self.predict_prob(x)   
        try:
            res_prob.shape[1]
            res = np.apply_along_axis(lambda x: [int(i) for i in x/x.max()],1,res_prob)
        except:
            res = np.array(np.round(res_prob))
        return (res)
    def save_model(self,path): #模型参数存储函数
        model_res = {}
        model_res['W'] = self.W
        model_res['K'] = self.K
        output = open(path+'.pkl', 'wb')
        pickle.dump(model_res, output)
        output.close()
    def load_model(self,path): #模型参数读取函数
        model_file = open(path+'.pkl', 'rb')
        model_res = pickle.load(model_file) 
        self.K = model_res['K']
        self.W = model_res['W']
        model_file.close()

#读取乳腺癌数据
cancer_data = load_breast_cancer()['data']
cancer_target = load_breast_cancer()['target']
cancer_data_train,cancer_data_test,cancer_target_train,cancer_target_test = train_test_split(cancer_data,cancer_target,test_size=0.2)

#乳腺癌测试数据二分类尝试
cancer_model = JQ_nnet() #实例化
cancer_model.fit(cancer_data_train,
                 cancer_target_train,
                 hidden=20,
                 iter=30000,
                 s=0.005,
                 error_iter=1000) #训练模型，迭代10000次，搜索步长为0.005
ann_pre_tr = cancer_model.predict_type(cancer_data_train) #训练集预测
ann_pre_te = cancer_model.predict_type(cancer_data_test) #测试集预测
accuracy_score(cancer_target_train,ann_pre_tr) #准确度：训练集
accuracy_score(cancer_target_test,ann_pre_te) #准确度：测试集

#读取mnist
import gzip #使用zlib来压缩和解压缩数据文件，读写gzip文件
import struct #通过引入struct模块来处理图片中的二进制数据

def read_data(label_path,image_path): #定义读取数据的函数
    with gzip.open(label_path) as flbl: #解压标签包
        magic, num = struct.unpack(">II",flbl.read(8)) #采用Big Endian的方式读取两个int类型的数据，且参考MNIST官方格式介绍，magic即为magic number (MSB first) 用于表示文件格式，num即为文件夹内包含的数据的数量
        label = np.fromstring(flbl.read(),dtype=np.int8) #将标签包中的每一个二进制数据转化成其对应的十进制数据，且转换后的数据格式为int8（-128 to 127）格式，返回一个数组
    with gzip.open(image_path,'rb') as fimg: #已只读形式解压图像包
        magic, num, rows, cols = struct.unpack(">IIII",fimg.read(16)) #采用Big Endian的方式读取四个int类型数据，且参考MNIST官方格式介绍，magic和num上同，rows和cols即表示图片的行数和列数
        image = np.fromstring(fimg.read(),dtype=np.uint8).reshape(len(label),rows,cols) #将图片包中的二进制数据读取后转换成无符号的int8格式的数组，并且以标签总个数，行数，列数重塑成一个新的多维数组
    return (label,image) #返回读取成功的label数组和image数组

(train_lbl, train_img) = read_data('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz') #构建训练数据
(val_lbl, val_img) = read_data('t10k-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz') #构建测试数据


def label_tsf(x): #多分类标签编码函数
    res = np.zeros([x.shape[0],np.unique(x).shape[0]])
    for i in range(x.shape[0]):
        res[i,x[i]]=1
    return res

np.random.seed(10) #随机种子
sample_idx =np.random.choice(60000,6000,replace=False) #随机抽取6000个样本
mnist_train_label = train_lbl[sample_idx]
mnist_train_y = label_tsf(mnist_train_label)
mnist_train_x = train_img[sample_idx,:,:].reshape([6000,784])
mnist_test_x = val_img.reshape([10000,784])
mnist_test_y = label_tsf(val_lbl)
#del train_lbl, train_img, val_lbl, val_img


#MNIST数据集多分类尝试
ann_model = JQ_nnet() #实例化
ann_model.fit(mnist_train_x,
              mnist_train_y,
              hidden=300, #隐层神经元
              iter=20000, #迭代数
              s=0.1, #初始步长
              decay=0.001, #衰减指数
              batch_size=100, #批处理数量
              loss='crossentropy', #多分类交叉熵损失函数
              error_iter=1000) #训练模型
ann_pre = ann_model.predict_prob(mnist_test_x)
accuracy_score(mnist_train_y,ann_model.predict_type(mnist_train_x)) #准确度：训练集
accuracy_score(mnist_test_y,ann_model.predict_type(mnist_test_x))#准确度：测试集
ann_model.save_model('nnet_model_mnist') #模型存储
ann_model.load_model('nnet_model_mnist') #模型加载


#keras模块神经网络建模
#对于keras模块的安装需要依赖于Theano或者Tensforflow框架，在Windows系统中安装较繁琐，因此下列步骤建议在Linux等系统中尝试
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras import optimizers

train_X = train_img.reshape([60000,784])
train_Y = label_tsf(train_lbl)
test_X = val_img.reshape([10000,784]) 
test_Y = val_lbl

net = Sequential() #实例化
net.add(Dropout(0.2,input_shape=(784,))) #过滤掉输入层20%的神经元
net.add(Dense(300,input_dim=784)) #输入层到隐藏层设置，输入层神经元为X数量，隐藏层神经元数10
net.add(Activation("sigmoid")) #输入层到隐藏层函数为sigmoid
net.add(Dropout(0.2)) #过滤掉20%的神经元
net.add(Dense(10,input_dim=300)) #隐藏层到输出层设置，上一隐藏层神经元为300，输出层神经元为10
net.add(Activation("softmax")) #隐藏层到输出层函数为softmax
sgd = optimizers.SGD(lr=0.01, decay=1e-6) #训练方法为随机梯度下降SGD
net.compile(loss='categorical_crossentropy', #损失函数binary_crossentropy
            optimizer=sgd,
            metrics=['accuracy']) #模型编译，目标为准确度
net.fit(train_X,train_Y,epochs=100,batch_size=100) #模型训练,设置迭代次数为1000次，批处理记录数32
net_predict = net.predict_classes(test_X) #预测类别
accuracy_score(train_lbl,net.predict_classes(train_X)) #预测准确度
accuracy_score(val_lbl,net_predict) #预测准确度

#对于keras模块神经网络建模的学习，可以参考keras中文文档，在学习神经网络乃至深度学习相关的算法上，keras对于初学者浅显易懂
#和当前较为流行的深度学习框架caffe、Tensorflow等相比，keras简洁，但在运算效率等方面则明显不如









