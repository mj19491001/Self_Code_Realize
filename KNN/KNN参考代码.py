# -*- coding: utf-8 -*-
"""
Created on Tue Jul 04 09:33:51 2017

@author: intern
"""

#对于要预测的数据，选择距离最近的K个训练集中的点，以其中的多数类别作为预测结果
import numpy as np

#数据集读取函数
def get_data(file_name,sep='\t'):
    data_open = open(file_name)
    lines = data_open.readlines()
    data = []
    for line in lines:
        line = line.replace('\n','')
        line = line.split(sep)
        a=[]
        for value in line:
            value = float(value)
            a.append(value)
        data.append(a)
    return np.array(data)
#训练集读取
horse_train = get_data(u'C:\\Users\\intern\\Desktop\\example\
\机器学习算法实现最终版\\horseColicTraining.txt')
horse_train_x = horse_train[:,range(horse_train.shape[1]-1)]
horse_train_y = horse_train[:,horse_train.shape[1]-1]
#测试集读取
horse_test = get_data(u'C:\\Users\\intern\\Desktop\\example\
\机器学习算法实现最终版\\horseColicTest.txt')
horse_test_x = horse_test[:,range(horse_test.shape[1]-1)]
horse_test_y = horse_test[:,horse_test.shape[1]-1]

#KNN模型定义
class knn_model():    
    def __init__(self,x,y): #在实例化时直接输入训练数据集
        self.train_data_x = np.array(x)
        self.train_data_y = np.array(y)
        self.train_data_x_std = (self.train_data_x - self.train_data_x.mean(axis=0))/self.train_data_x.std(axis=0)
        self.train_data_x_01 = (self.train_data_x - self.train_data_x.min(axis=0))/(self.train_data_x.max(axis=0)-self.train_data_x.min(axis=0))
    def standard_zeromean(self,x): #0-1标准化函数
        return(x - self.train_data_x.mean(axis=0))/self.train_data_x.std(axis=0)
    def standard_01(self,x): #标准化函数
        return(x - self.train_data_x.min(axis=0))/(self.train_data_x.max(axis=0)-self.train_data_x.min(axis=0))        
    def predict_1line(self,x,k,train_x,train_y): #对一行数据的预测
        distance = ((train_x-x)**2).sum(axis=1)
        mink_dis = np.sort(distance)[:k]
        def isin(a):
            return(a in mink_dis)
        distance = distance.reshape(distance.shape[0],1)
        sel_lab=np.apply_along_axis(isin,1,distance)
        type_pre = np.unique(train_y[sel_lab])[0]
        for i in range(1,len(np.unique(train_y[sel_lab]))):
            if sum(train_y[sel_lab] == np.unique(train_y[sel_lab])[i]) >  sum(train_y[sel_lab] == np.unique(train_y[sel_lab])[i-1]):
                type_pre = np.unique(train_y[sel_lab])[i]
        return(type_pre)
    def predict(self,x,k=10,std=None): #预测函数
        type_pred = []
        if std == "zeromean":
            x = self.standard_zeromean(x)
            for i in range(x.shape[0]):
                type_pred.append(self.predict_1line(x[i,:],k,self.train_data_x_std,self.train_data_y))
        elif std == "0-1":
            x = self.standard_01(x)
            for i in range(x.shape[0]):
                type_pred.append(self.predict_1line(x[i,:],k,self.train_data_x_01,self.train_data_y))
        else:
            for i in range(x.shape[0]):
                type_pred.append(self.predict_1line(x[i,:],k,self.train_data_x,self.train_data_y))
        return(np.array(type_pred))
    def Accuracy(self,x,y,k=10,std=None): #预测准确度的评价
        y_pre = self.predict(x,k=k,std=std)
        Accuracy = sum(y_pre == y)/float(len(y))
        return(Accuracy)

#采用之前的horse数据集进行测试        
knn_horse = knn_model(horse_train_x,horse_train_y) #将训练数据集的自变量和目标变量输入模型
#设置近邻数为5，未标准化的预测结果
print 'the Accuracy of KNN without data standardization is %s' % knn_horse.Accuracy(horse_test_x,horse_test_y,k=5)
#设置近邻数为5，对数据进行0-1标准化再用于预测
print 'the Accuracy of KNN with data 0-1 standardization is %s' % knn_horse.Accuracy(horse_test_x,horse_test_y,k=5,std='0-1') 

#sklearn模块knn选择最佳参数K建模示例
from sklearn.preprocessing import StandardScaler #模块：标准化函数(效果一般)
from sklearn.preprocessing import MinMaxScaler #模块：01标准化函数 
from sklearn.neighbors import KNeighborsClassifier #模块：KNN分类模型
from sklearn.pipeline import Pipeline #管道串联模块
from sklearn.grid_search import GridSearchCV #网格搜索模块

#管道串联标准化步骤和建模步骤（暂时选用01标准化）
pipe_process = Pipeline([('scale_m',MinMaxScaler()),('knn_sklearn',KNeighborsClassifier())])

#传入KNN建模参数n_neighbors，候选值的范围包含了整数3-20
parameters = {'knn_sklearn__n_neighbors':np.array(range(3,21))}

#网格参数搜索，输入之前的模型流程pipe_process,候选参数parameters，并且设置5折交叉验证
gs = GridSearchCV(pipe_process,parameters,verbose=2,refit=True,cv=5) #设置备选参数组

gs.fit(horse_train_x,horse_train_y) #模型训练过程
print gs.best_params_,gs.best_score_ #查看最佳参数和评分（准确度）

#最佳参数的KNN建模对预测数据预测效果）
print 'The Accuracy of KNeighborsClassifier model with best parameter and MinMaxScaler is',gs.score(horse_test_x,horse_test_y)














