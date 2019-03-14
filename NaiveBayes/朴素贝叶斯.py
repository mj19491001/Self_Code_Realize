#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:03:47 2017

@author: JQstyle
"""
import numpy as np
import pandas as pd #用于加载数据集
horse = pd.read_table(u'C:/Users/JQstyle/Desktop/杂事整理/技术交流/公众号文章/朴素贝叶斯/horseColicTraining.txt',
                  sep='\t',names=['x' + str(i) for i in range(21)]+['y'])
horse_t = pd.read_table(u'C:/Users/JQstyle/Desktop/杂事整理/技术交流/公众号文章/朴素贝叶斯/horseColicTest.txt',
                  sep='\t',names=['x' + str(i) for i in range(21)]+['y'])

#分箱器
def bin_get(x,your_bins):
    new_x = x.copy()
    for i in range(your_bins.shape[0]):
        if i==0:
            new_x[x<=your_bins[i]]=0
        if i>0:
            new_x[(x>your_bins[i-1])&(x<=your_bins[i])]=i
        if i==your_bins.shape[0]-1:
            new_x[x>your_bins[i]]=(i+1)
    return new_x
#分箱
horse_bin = horse.copy()
horse_t_bin = horse_t.copy()
for i in range(horse.shape[1]-1):
    if len(set(horse_bin.ix[:,i])) > 7:#只对拥有7个unique值以上的变量进行分箱工作
        horse_bin.iloc[:,i] = bin_get(horse.ix[:,i],
                      np.array(horse.ix[:,i].quantile([0.2,0.4,0.6,0.8])))
        horse_t_bin.iloc[:,i] = bin_get(horse_t.ix[:,i],
                      np.array(horse.ix[:,i].quantile([0.2,0.4,0.6,0.8])))
#转化为numpy矩阵便于建模
horse_bin = np.array(horse_bin)
horse_t_bin = np.array(horse_t_bin)

#朴素贝叶斯模型
#条件概率独立性假设：P(X|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)。。。
#对于已知X的Y的概率计算：P(Yi|X) = P(X,Y)/P(X) = P(X|Yi)*P(Yi)/P(X) = (P(x1|Yi)*P(x2|Yi)*P(x3|Yi)..)*P(Yi)/Σ((P(x1|Yi)*P(x2|Yi)*P(x3|Yi)..)*P(Yi))
class Naive_Bayes_JQ:
    def pri_prob_for1(self,x,lapras=0): #对先验概率的计算函数
        prob_dir1 = {}
        for i in set(x):
            prob_dir1[i] = float(sum(x == i)+lapras)/(len(x)+lapras*len(set(x)))
        return(prob_dir1)
    def pri_prob_for2(self,x,y,lapras=0):  #先验概率的计算函数（X且Y）
        prob_dir2 = {}
        for i in set(x):
            for j in set(y):
                prob_dir2[(i,j)] = float(sum(y[x==i]==j)+lapras)/(len(x)+lapras*len(set(x)))
        return(prob_dir2)
    def con_prob(self,x,y,lapras=0):  #条件概率的计算函数（Y到X）
        prob_dir3 = {}
        pxy = self.pri_prob_for2(x,y,lapras)
        py = self.pri_prob_for1(y,lapras)
        for i in set(x):
            for j in set(y):
                if lapras !=0:
                    n=0
                    for k in range(len(y)):
                        n += ((y[k]==j) and (x[k]==i))
                    prob_dir3[(i,j)] = float(n+lapras)/(sum(y==j)+lapras*len(set(x)))
                else:
                    prob_dir3[(i,j)] = pxy[(i,j)]/py[j]
        return(prob_dir3)        
    def Naive_Bayes_fit(self,x,y,lapras=0):  #模型训练，得到所有Y的先验概率和Y到X各类的条件概率
        self.pri_prob_y = self.pri_prob_for1(y,lapras=lapras)
        self.con_prob_xy = {}
        for i in range(x.shape[1]):
            self.con_prob_xy[i] = self.con_prob(x[:,i],y,lapras=lapras)
    def predict_line(self,x,prob=False):  #单行预测函数
        tar_pro = {}
        for i in range(len(self.pri_prob_y)):
            tar_pro[self.pri_prob_y.keys()[i]] = self.pri_prob_y[i]
            for j in range(x.shape[0]):
                tar_pro[self.pri_prob_y.keys()[i]] = tar_pro[self.pri_prob_y.keys()[i]]*self.con_prob_xy[j][(x[j],self.pri_prob_y.keys()[i])]
        if prob:
            prob_sum = sum(np.array(tar_pro.values()))
            for k in tar_pro.keys():
                tar_pro[k] = tar_pro[k]/prob_sum
            return(tar_pro)                            
        type_pre = np.array(tar_pro.keys())[np.array(tar_pro.values())==np.array(max(tar_pro.values()))][0]
        return(type_pre)
    def Naive_Bayes_predict(self,x,prob=False):  #预测函数
        y_pre = []
        for i in range(x.shape[0]):
            y_pre.append (self.predict_line(x[i,:],prob))
        return(np.array(y_pre))
    def Accuracy_score(self,x,y):
        y_pre = self.Naive_Bayes_predict(x)
        Accuracy_value = sum(y_pre==y)/float(y.shape[0])
        return Accuracy_value
        
#测试数据示例
model2 = Naive_Bayes_JQ() #实例化模型
model2.Naive_Bayes_fit(horse_bin[:,:21],horse_bin[:,21],lapras=1) #模型训练（采用拉普拉斯平滑）
model2.Naive_Bayes_predict(horse_t_bin[:,:21],prob=True) #输出概率
model2.Accuracy_score(horse_t_bin[:,:21],horse_t_bin[:,21]) #测试集准确度




















