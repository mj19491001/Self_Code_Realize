#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 13:09:36 2017

@author: jq_tongdun
"""

import numpy as np
import pandas as pd

horse = pd.read_table('/Users/jq_tongdun/Desktop/tech_learning/horseColicTraining.txt',
                  sep='\t',names=['x' + str(i) for i in range(21)]+['y'])
horse_t = pd.read_table('/Users/jq_tongdun/Desktop/tech_learning/horseColicTest.txt',
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

#分箱:将horse数据集的每个数值型字段等分为三箱用于之后的ID3算法建模
horse_bin = horse.copy()
horse_t_bin = horse_t.copy()
for i in range(horse.shape[1]-1):
    if len(set(horse_bin.ix[:,i])) > 3:
        horse_bin.iloc[:,i] = bin_get(horse.ix[:,i],
                      np.array(horse.ix[:,i].quantile([1/3.,1/6.])))
        horse_t_bin.iloc[:,i] = bin_get(horse_t.ix[:,i],
                      np.array(horse.ix[:,i].quantile([1/3.,1/6.])))

horse_bin = np.array(horse_bin)
horse_t_bin = np.array(horse_t_bin)

#训练集和测试集
horse_train_x = np.array(horse.ix[:,range(21)])
horse_train_y = np.array(horse.ix[:,21])
horse_test_x = np.array(horse_t.ix[:,range(21)])
horse_test_y = np.array(horse_t.ix[:,21])

horse_bin_train = horse_bin[:,range(21)]
horse_bin_test = horse_t_bin[:,range(21)]

#ID3算法采用的是信息增益筛选分类变量
#信息熵： H(X) = -Σpi*LOG2(pi)  pi为x各个类别概率
#条件熵： H(Y|X) = Σpi*H(Y|X=xi)  pi为x各个类别概率，后一项是相同类别X对应的Y部分的信息熵
#信息增益（互信息）: G(Y,X) = H（Y） - H(Y|X)
#信息增益率（C45算法采用）： Gr(Y,X) = G(Y,X)/H(X)
##伪代码流程
'''
输入目标Y，X，输出决策树T
1.停止条件：若Y属于同一类Ck（或超过阈值大部分Y属于同一类），则此时为单节点树
2.停止条件：若X为∅，即没有X可供分类，则此时为叶节点，输出Y的多数类Ck
3.计算Y和各个X的信息增益G(Y,X)，设最大的特征X为Xg
4.若G（Y,Xg）小于阈值e，则输出Y的多数类Ck
5.否则，将Y按照Xg各个类别划分非空子集Yi
6.对子节点的各个Yi和对应的X递归1-5步
'''
#ID3算法
class desicion_tree: #信息熵
    import numpy as np
    ID3_tree_model = {}
    def inf_ent(self,x):#信息熵
        inf_ent = 0
        for type in set(x):
            pi = float(sum(x==type))/x.shape[0]
            inf_ent -= pi*np.log2(pi)
        return (inf_ent)
    def con_ent(self,y,x):  #条件熵
        con_ent = 0
        for type in set(x):
            yt = y[x==type]
            pi = float(sum(x==type))/x.shape[0]
            con_ent += pi*self.inf_ent(yt)
        return(con_ent)
    def inf_gain(self,y,x):  #信息增益
        inf_gain = self.inf_ent(y)-self.con_ent(y,x)
        return(inf_gain)
    def des_ID3(self,x,y,x_name,stop_p=0.95,inf_min=0.1,thre_num=5):  #ID3递归建模函数
        ID3_tree = {}
        max_rate = 0
        y_res = 0
        for tp in set(y):
            if sum(y==tp)/float(y.shape[0]) > max_rate:
                max_rate = sum(y==tp)/float(y.shape[0])
                y_res = tp
        if len(y) <= thre_num: #停止条件，当前叶节点Y记录数低于阈值时
            return(y_res)
        if max_rate >= stop_p : #停止条件，当前叶节点Y的众数比例超过阈值时  
            return(y_res)
        inf_gain = np.zeros(x.shape[1])
        for i in range(x.shape[1]): #计算各个特征对Y信息增益
            inf_gain[i] = self.inf_gain(y,x[:,i])
        sel_name = x_name[inf_gain == inf_gain.max()][0] #选择最大的信息增益的特征
        x_name_new = x_name[inf_gain != inf_gain.max()]
        for tp in set(x[:,inf_gain == inf_gain.max()][:,0]): #对选定特征的每一个取值下的子节点进行递归
            y_new = y[x[:,inf_gain == inf_gain.max()][:,0]==tp]
            x_new = x[x[:,inf_gain == inf_gain.max()][:,0]==tp,:]
            x_new = x_new[:,inf_gain != inf_gain.max()]
            ID3_tree[(sel_name, tp)] = self.des_ID3(x_new,y_new,x_name_new,stop_p=stop_p,inf_min=inf_min,thre_num=thre_num)
        return(ID3_tree)
    def des_ID3_fit(self,x,y,x_name,stop_p=0.95,inf_min=0.1,thre_num=5): #ID3最终使用的建模函数
        self.ID3_tree_model = self.des_ID3(x,y,x_name,stop_p=stop_p,inf_min=inf_min,thre_num=thre_num)
        return(self.ID3_tree_model)
    def ID3_predict_rec(self,x,x_name,model):
        name = x_name[x_name == model.keys()[0][0]][0]
        model = model[(name,x[x_name == model.keys()[0][0]][0])]
        if isinstance(model,dict):
            return(self.ID3_predict_rec(x,x_name,model=model))
        else:
            return(model)
    def ID3_predict_line(self,x,x_name):  #单条记录的预测函数
        result = self.ID3_predict_rec(x,x_name,self.ID3_tree_model)
        return(result)
    def ID3_predict(self,x,x_name):  #数据集预测函数
        if self.ID3_tree_model == {}:
            print("the model need to be trained")
        else:
            y_pre = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                y_pre[i] = self.ID3_predict_line(x[i,:],x_name)
            return(y_pre)  
        
#ID3算法的测试数据集    
example_x_name = np.array(["Age","Job","House","Credit"])
example_x = np.array([["youth","youth","youth","youth","youth","middle","middle",
                       "middle","middle","middle","old","old","old","old","old"],
                      ['no','no','yes','yes','no','no','no','yes','no','no','no',
                       'no','yes','yes','no'],
                      ['no','no','no','yes','no','no','no','yes','yes','yes',
                       'yes','yes','no','no','no'],
                      ['common','good','good','common','common','common','good',
                       'good','very good','very good','very good','good','good',
                       'very good','common']]).T
example_y = np.array([0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])

#ID3算法测试
ID3_model=desicion_tree()
ID3_model.des_ID3_fit(x=example_x,y=example_y,x_name=example_x_name) #训练模型
ID3_model.ID3_tree_model #查看ID3决策树结果 
ID3_model.ID3_predict(example_x,example_x_name) #对数据集的类别预测(回带)

#利用ID31对分箱以后的horse数据集进行测试
horse_name = np.array(['X%s' %i for i in range(21) ]) #为X字段命名
ID3_horse=desicion_tree()
ID3_horse.des_ID3_fit(x=horse_bin_train,y=horse_train_y,x_name=horse_name,stop_p=0.8,thre_num=10) #训练模型
sum(ID3_horse.ID3_predict(x=horse_bin_test,x_name=horse_name)==horse_test_y)/float(len(horse_test_y)) 

#附：cart树对于连续型自变量的处理
#基尼系数：Gini（x） = Σpi（1-pi）  pi为X中各个类的占比，cart一般均分为2类
#Gini（Y，X） = Σ (xi/N * Gini(Y|X=xi))
'''
cart分类二叉树建模流程：
输入：X,Y
1.停止条件：当前Y只剩一类时；所有X均以被作为节点之后等等；此时将当前的y的众数作为叶节点类
2.基于基尼系数最小化的规则对每一个X字段选取最优切分点
3.将切分以后的X中的最小基尼系数的字段作为本次的分类节点
4.将Y根据选取的字段进行二分类
5.对每个分类的Y的对应的剩余字段X进行2,3,4的递归
'''

class cart_tree:
    import numpy as np
    cart_tree = {}
    def Gini(self,x,y): #基尼系数计算函数
        Gini = 0
        for i in np.unique(x):
            p1 = sum(x==i)/float(x.shape[0])         
            for j in np.unique(y):
                p2 = sum(y[x==i]==j)/float(y[x==i].shape[0])
                Gini += p1 * p2 * (1 - p2)
        return(Gini)
    def break_point_Gini(self,x,y): #对单字段的切分点确定
        break_point = x[0]
        Gini_xy = self.Gini(x<=x[0],y)
        for i in range(1,x.shape[0]):
            if self.Gini(x<=x[i],y) < Gini_xy:
                Gini_xy = self.Gini(x<=x[i],y)
                break_point = x[i]
        return break_point
    def new_cut_x(self,x,y): #对训练数据集X的分箱并且输出切分点
        new_x = np.zeros(x.shape)
        break_point = []
        for i in range(x.shape[1]):
            bk = self.break_point_Gini(x[:,i],y)
            new_x[:,i] = x[:,i] <= bk
            break_point.append(bk)
        break_point = np.array(break_point)
        return new_x,break_point
    def cart_tree_fit(self,x,y,x_names,thre_type_rate=0.8,thre_num=5): #模型的训练函数
        tree = {}
        print x.shape
        for i in np.unique(y):
            if sum(y==i)/float(y.shape[0]) >= thre_type_rate: #停止条件：当Y中同一类的比重超过阈值
                return(i)
            if sum(y==i) <= thre_num: #停止条件：当Y的数量低于阈值
                return(i)
        if x.shape[0] == 0: #停止条件：当所有的X均已被作为分类字段
            type_y = np.unique(y)[0]
            for i in np.unique(y)[1:]:
                if sum(y==i)/float(y.shape[0]) > sum(y==type_y)/float(y.shape[0]):
                    type_y = i
            return(type_y)
        sel_x = 0
        if len(x.shape) > 1: #当X字段数多于2时
            new_cut_x, break_point = self.new_cut_x(x,y) #提取各个X字段切分点和分享以后的X
            gini_min = self.Gini(x[:,0],y)
            for j in range(1,new_cut_x.shape[1]): #找出基尼系数最小的X字段
                if self.Gini(new_cut_x[:,j],y) < gini_min:
                    gini_min = self.Gini(new_cut_x[:,j],y)
                    sel_x = j  
            new_x_low = x[x[:,sel_x]<=break_point[sel_x],:] 
            new_x_low = new_x_low[:,np.array(range(x.shape[1]))!=sel_x] #当前字段低于切分点的剩余X数据集
            new_x_high = x[x[:,sel_x]>break_point[sel_x],:]
            new_x_high = new_x_high[:,np.array(range(x.shape[1]))!=sel_x] #当前字段高于切分点的剩余X数据集
            new_y_low = y[x[:,sel_x]<=break_point[sel_x]] #当前字段低于切分点的Y数据集
            new_y_high = y[x[:,sel_x]>break_point[sel_x]] #当前字段高于切分点的Y数据集
            names_new = x_names[np.array(range(x_names.shape[0]))!=sel_x]
            label_low = x_names[sel_x] +'<=%s' %break_point[sel_x] #节点标签1
            label_high = x_names[sel_x] +'>%s' %break_point[sel_x] #节点标签2
        tree[label_low] = self.cart_tree_fit(new_x_low,new_y_low,names_new,thre_type_rate,thre_num) #子节点递归1
        tree[label_high] = self.cart_tree_fit(new_x_high,new_y_high,names_new,thre_type_rate,thre_num) #子节点递归2
        if tree[label_low] == tree[label_high]:
            return tree[label_high]
        self.cart_tree = tree
        return self.cart_tree
    def cart_predict_line(self,x,x_names,model): #单行预测函数
        import re
        if isinstance(model,dict):
            sel_x = x[x_names==re.split("<=|>",model.keys()[0])[0]]           
            bp = re.split("<=|>",model.keys()[0])[1]
            if sel_x <= float(bp):
                key_x = x_names[x_names==re.split("<=|>",model.keys()[0])[0]][0] + "<=" + bp
            else:
                key_x = x_names[x_names==re.split("<=|>",model.keys()[0])[0]][0] + ">" + bp
            return self.cart_predict_line(x,x_names,model[key_x])
        else:
            return model
    def cart_predict(self,x,x_names): #预测函数
        if self.cart_tree == {}:
            return "Please fit the model"
        else:
            result = []
            for i in range(x.shape[0]):
                result.append(self.cart_predict_line(x[i,:],x_names,self.cart_tree))
            result = np.array(result)
            return result
        
#基于之前的horse数据集测试      
cart_model=cart_tree() #实例化
cart_model.cart_tree_fit(horse_train_x,horse_train_y,horse_name,thre_num=5) #模型训练
cart_model.cart_tree #查看决策树的内容
cart_model_pre = cart_model.cart_predict(horse_test_x,horse_name) #预测
sum(cart_model_pre == horse_test_y)/float(len(horse_test_y)) #准确度

#采用sklearn模块构建cart决策树
from sklearn import tree
clf_cart =tree.DecisionTreeClassifier(criterion='gini',max_depth = 5,min_samples_leaf=5) #实例化，区分一个内部节点需要的最少的样本数为10，一个叶节点最小样本数为5
clf_cart.fit(horse_train_x,horse_train_y) #模型训练
clf_cart_pre = clf_cart.predict(horse_test_x) #预测
sum(clf_cart_pre == horse_test_y)/float(len(horse_test_y)) #准确度

#绘制决策树
import pydotplus 
from sklearn.tree import export_graphviz
from IPython.display import Image  
with open(r"/Users/jq_tongdun/Desktop/tree.dot",'w') as f: #保存决策树模型
    f = export_graphviz(clf_cart, feature_names = horse_name, out_file = f)
graph = pydotplus.graphviz.graph_from_dot_file(r"/Users/jq_tongdun/Desktop/tree.dot") 
#graph.write_pdf(r"/Users/jq_tongdun/Desktop/tree.pdf") #创建PDF存储决策树图
Image(graph.create_png()) #直接绘制















