1#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:18:53 2017

@author: jq_tongdun
"""

import numpy as np
import pandas as pd
import random
import multiprocessing #多线程提升效率

horse = pd.read_table(u'C:/Users/JQstyle/Desktop/杂事整理/技术交流/公众号文章/朴素贝叶斯/horseColicTraining.txt',
                  sep='\t',names=['x' + str(i) for i in range(21)]+['y'])
horse_t = pd.read_table(u'C:/Users/JQstyle/Desktop/杂事整理/技术交流/公众号文章/朴素贝叶斯/horseColicTest.txt',
                  sep='\t',names=['x' + str(i) for i in range(21)]+['y'])

horse = np.array(horse)
horse_t = np.array(horse_t)

horse_train_x,horse_train_y = horse[:,range(21)],horse[:,21]
horse_test_x,horse_test_y = horse_t[:,range(21)],horse_t[:,21]
horse_name = np.array(['X%s' %i for i in range(21) ]) #为X字段命名

#基础cart分类树
class cart_classify:
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
    def cart_tree_fit(self,x,y,x_names,thre_num=5,max_depth=3,now_depth=0): #模型的训练函数
        tree = {}
        type_res = np.unique(y)[0]
        max_num = sum(y==np.unique(y)[0])
        if np.unique(y).shape[0] < 2:
            return y[0]
        for type0 in np.unique(y)[1:]:
            if max_num < sum(y==type0):
                type_res = type0
                max_num = sum(y==type0)        
        if y.shape[0] <= thre_num: #停止条件：当Y的数量低于阈值
            return type_res
        if now_depth >= max_depth: #停止条件：当所有的X均已被作为分类字段
            return type_res
        sel_x = 0
        new_cut_x, break_point = self.new_cut_x(x,y) #提取各个X字段切分点和分享以后的X
        gini_min = self.Gini(x[:,0],y)
        for j in range(1,new_cut_x.shape[1]): #找出基尼系数最小的X字段
            if self.Gini(new_cut_x[:,j],y) < gini_min:
                gini_min = self.Gini(new_cut_x[:,j],y)
                sel_x = j  
        new_x_low = x[x[:,sel_x]<=break_point[sel_x],:]  #当前字段低于切分点的剩余X数据集
        new_x_high = x[x[:,sel_x]>break_point[sel_x],:]  #当前字段高于切分点的剩余X数据集
        new_y_low = y[x[:,sel_x]<=break_point[sel_x]] #当前字段低于切分点的Y数据集
        new_y_high = y[x[:,sel_x]>break_point[sel_x]] #当前字段高于切分点的Y数据集
        label_low = x_names[sel_x] +'<=%s' %break_point[sel_x] #节点标签1
        label_high = x_names[sel_x] +'>%s' %break_point[sel_x] #节点标签2
        tree[label_low] = self.cart_tree_fit(new_x_low,new_y_low,x_names,thre_num,max_depth,now_depth+1) #子节点递归1
        tree[label_high] = self.cart_tree_fit(new_x_high,new_y_high,x_names,thre_num,max_depth,now_depth+1) #子节点递归2
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

ex = cart_classify()
ex.cart_tree_fit(horse_train_x,horse_train_y,horse_name,max_depth=5)
sum(ex.cart_predict(horse_test_x,horse_name)==horse_test_y)/float(horse_test_y.shape[0])

#随机森林尝试
class RandomForest(cart_classify):
    tree_forest = []
    tree_names = []
    def cart_tree_fit(self,x,y,x_names,thre_num=5,max_depth=3,
                      now_depth=0): #重写cart树训练函数
        tree = {}
        if np.unique(y).shape[0] < 2:
            return y[0]
        type_res = np.unique(y)[0]
        max_num = sum(y==np.unique(y)[0])
        for type0 in np.unique(y)[1:]:
            if max_num < sum(y==type0):
                type_res = type0
                max_num = sum(y==type0)        
        if y.shape[0] <= thre_num: #停止条件：当Y的数量低于阈值
            return type_res
        if now_depth >= max_depth: #停止条件：当所有的X均已被作为分类字段
            return type_res
        sel_x = 0
        new_cut_x, break_point = self.new_cut_x(x,y) #提取各个X字段切分点和分享以后的X
        gini_min = self.Gini(x[:,0],y)
        for j in range(1,new_cut_x.shape[1]): #找出基尼系数最小的X字段
            if self.Gini(new_cut_x[:,j],y) < gini_min:
                gini_min = self.Gini(new_cut_x[:,j],y)
                sel_x = j  
        new_x_low = x[x[:,sel_x]<=break_point[sel_x],:]  #当前字段低于切分点的剩余X数据集
        new_x_high = x[x[:,sel_x]>break_point[sel_x],:]  #当前字段高于切分点的剩余X数据集
        new_y_low = y[x[:,sel_x]<=break_point[sel_x]] #当前字段低于切分点的Y数据集
        new_y_high = y[x[:,sel_x]>break_point[sel_x]] #当前字段高于切分点的Y数据集
        label_low = x_names[sel_x] +'<=%s' %break_point[sel_x] #节点标签1
        label_high = x_names[sel_x] +'>%s' %break_point[sel_x] #节点标签2
        if np.unique(new_y_low).shape[0]<2 or np.unique(new_y_high).shape[0]<2:
            return type_res        
        tree[label_low] = self.cart_tree_fit(new_x_low,new_y_low,x_names,thre_num,max_depth,now_depth+1) #子节点递归1
        tree[label_high] = self.cart_tree_fit(new_x_high,new_y_high,x_names,thre_num,max_depth,now_depth+1) #子节点递归2
        if tree[label_low] == tree[label_high]:
            return tree[label_high]
        return tree 
    def cart_predict(self,x,x_names,model): #预测函数
        result = []
        for i in range(x.shape[0]):
            result.append(self.cart_predict_line(x[i,:],x_names,model))
        result = np.array(result)
        return result
    def random_sel_mod(self,x,y,x_names,max_depth,row_samples,thre_num,
                       seedc,seedr): #单棵决策树的训练函数(基于随机特征和记录)
        random.seed(seedc)
        col_sel = random.sample(range(x.shape[1]),max_depth)
        random.seed(seedr)
        row_sel = random.sample(range(y.shape[0]),int(round(y.shape[0]*row_samples)))
        x_tmp = x[row_sel,:][:,col_sel]
        y_tmp = y[row_sel]
        names_tmp = x_names[col_sel]
        tree_tmp = self.cart_tree_fit(x_tmp,y_tmp,names_tmp,thre_num=thre_num,max_depth=max_depth)
        return tree_tmp,names_tmp
    def RandomForest_fit(self,x,y,x_names,num_trees=9,max_depth=None,
                         row_samples=0.3,thre_num=5,seed=100): #随机森林建模主函数
        if max_depth == None:
            max_depth == round(x.shape[1])
        self.tree_names = []
        self.tree_forest = []
        for i in range(num_trees):
            tree_tmp,names_tmp = self.random_sel_mod(x,y,x_names,max_depth,row_samples,thre_num,seed+i,2*seed+i)
            self.tree_names.append(names_tmp)
            self.tree_forest.append(tree_tmp)

    def RandomForest_predict(self,x,x_names): #预测函数：输出概率
        result = np.zeros(x.shape[0])
        for i in range(len(self.tree_forest)):
            pre_tmp = self.cart_predict(x,x_names,self.tree_forest[i])
            result = result + pre_tmp
        result = result/float(len(self.tree_forest))
        return result
    def RandomForest_predict_type(self,x,x_names): #预测函数：输出类别
        return np.round(self.RandomForest_predict(x,x_names))
    def Acc_Score(self,x,x_names,y): #准确度评价函数
        pre_result = self.RandomForest_predict_type(x,x_names)
        acc_value = sum(pre_result==y)/float(y.shape[0])
        return acc_value
    def __call__(self,x,y,x_names,max_depth,row_samples,thre_num,
                 seedc,seedr): #定义类默认函数，用于多进程优化
        return self.random_sel_mod(x,y,x_names,max_depth,row_samples,thre_num,seedc,seedr)
    def RandomForest_fit_process(self,x,y,x_names,num_trees=9,max_depth=None,
                                 row_samples=0.3,thre_num=5,seed=100,
                                 process=2): #多进程优化的随机森林训练函数
        if max_depth == None:
            max_depth == round(x.shape[1])
        self.tree_names = []
        self.tree_forest = []
        pool = multiprocessing.Pool(processes=process)
        result = []
        for i in range(num_trees):
            result.append(pool.apply_async(self,(x,y,x_names,max_depth,row_samples,thre_num,
                                   seed+i,2*seed+i)))
        pool.close()
        pool.join()
        for res in result:
            res_tmp = res.get()
            self.tree_names.append(res_tmp[1])
            self.tree_forest.append(res_tmp[0])

#horse数据集           
RF_model = RandomForest()
RF_model.RandomForest_fit(horse_train_x,horse_train_y,horse_name,num_trees=20,
                          max_depth=5,seed=1000)       
RF_model.Acc_Score(horse_test_x,horse_name,horse_test_y) #预测准确度

#sklearn库的随机森林示例
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators=100)
RF_model.fit(horse_train_x,horse_train_y)
y_pre_RF=RF_model.predict(horse_test_x) #预测模型
sum(y_pre_RF==horse_test_y)/float(horse_test_y.shape[0]) #准确度

RF_model.feature_importances_ #查看变量的重要性

#超参数搜索寻找最优模型
from sklearn.grid_search import GridSearchCV #网格搜索模块

clf = RandomForestClassifier()
#候选参数：树的数量，最大树深，选择的变量树
parameters = {'n_estimators':np.array([25,50,100]),'max_depth':np.array([2,3,4,5,6,7,8]),'max_features':np.array([4,5,6,7])}
#网格参数搜索，输入之前的模型流程pipe_process,候选参数parameters，并且设置5折交叉验证
gs_RF = GridSearchCV(clf,parameters,verbose=2,refit=True,cv=5) #设置备选参数组
gs_RF.fit(horse_train_x,horse_train_y) #模型训练过程
print gs_RF.best_params_,gs_RF.best_score_ #查看最佳参数和评分（准确度）
#最佳参数的KNN建模对预测数据预测效果）
print 'The Accuracy of GradientBoostingClassifier model with best parameter and MinMaxScaler is',gs_RF.score(horse_test_x,horse_test_y)

#对比RF原函数和优化以后的函数的建模时间：
import time
#原函数所花时间
start = time.time()
RF_model = RandomForest()
RF_model.RandomForest_fit(horse_train_x,horse_train_y,horse_name,num_trees=20,
                          max_depth=5,seed=1000)       
end = time.time()
print end - start
#采用多进程优化以后的RF建模时间(核心数process可以根据硬件条件确定，默认为双核)
start = time.time()
RF_model = RandomForest() 
RF_model.RandomForest_fit_process(horse_train_x,horse_train_y,horse_name,
                                  num_trees=20,max_depth=5,seed=1000,process=2)       
end = time.time()
print end - start














