#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:01:46 2017

@author: jq_tongdun
"""
import numpy as np
import pandas as pd

horse = pd.read_table('/Users/jq_tongdun/Desktop/tech_learning/horseColicTraining.txt',
                  sep='\t',names=['x' + str(i) for i in range(21)]+['y'])
horse_t = pd.read_table('/Users/jq_tongdun/Desktop/tech_learning/horseColicTest.txt',
                  sep='\t',names=['x' + str(i) for i in range(21)]+['y'])

horse = np.array(horse)
horse_t = np.array(horse_t)

horse_train_x,horse_train_y = horse[:,range(21)],horse[:,21]
horse_test_x,horse_test_y = horse_t[:,range(21)],horse_t[:,21]
horse_name = np.array(['X%s' %i for i in range(21) ]) #为X字段命名
#基础cart回归树
class cart_regression:
    import numpy as np
    cart_tree = {}
    def res_calc(self,x,y):#误差平方和函数
        res_value = 0
        for i in np.unique(x):
            ci = np.mean(y[x==i])
            res_value += sum((y[x==i]-ci)**2)
        return res_value
    def break_point_res(self,x,y): #对单字段的切分点确定
        break_point = x[0]
        res_xy = self.res_calc(x<=x[0],y)
        for i in range(1,x.shape[0]):
            if self.res_calc(x<=x[i],y) < res_xy:
                res_xy = self.res_calc(x<=x[i],y)
                break_point = x[i]
        return break_point
    def new_cut_x(self,x,y): #对训练数据集X的分箱并且输出切分点
        new_x = np.zeros(x.shape)
        break_point = []
        for i in range(x.shape[1]):
            bk = self.break_point_res(x[:,i],y)
            new_x[:,i] = x[:,i] <= bk
            break_point.append(bk)
        break_point = np.array(break_point)
        return new_x,break_point
    def cart_tree_fit(self,x,y,x_names,thre_num=5,
                      max_depth=10,now_depth=0): #模型的训练函数
        tree = {}
        y_res = np.average(y)
        if y.shape[0] <= thre_num: #停止条件：当Y的数量低于阈值
            return y_res
        if now_depth >= max_depth:
            return y_res
        if x.shape[0] == 0: #停止条件：当所有的X均已被作为分类字段
            return y_res
        sel_x = 0
        new_cut_x, break_point = self.new_cut_x(x,y) #提取各个X字段切分点和分享以后的X
        gini_min = self.res_calc(x[:,0],y)
        for j in range(1,new_cut_x.shape[1]): #找出最小的X字段
            if self.res_calc(new_cut_x[:,j],y) < gini_min:
                gini_min = self.res_calc(new_cut_x[:,j],y)
                sel_x = j  
        new_x_low = x[x[:,sel_x]<=break_point[sel_x],:] 
        #new_x_low = new_x_low[:,np.array(range(x.shape[1]))!=sel_x] #当前字段低于切分点的剩余X数据集
        new_x_high = x[x[:,sel_x]>break_point[sel_x],:]
        #new_x_high = new_x_high[:,np.array(range(x.shape[1]))!=sel_x] #当前字段高于切分点的剩余X数据集
        new_y_low = y[x[:,sel_x]<=break_point[sel_x]] #当前字段低于切分点的Y数据集
        new_y_high = y[x[:,sel_x]>break_point[sel_x]] #当前字段高于切分点的Y数据集
        names_new = x_names#[np.array(range(x_names.shape[0]))!=sel_x]
        label_low = x_names[sel_x] +'<=%s' %break_point[sel_x] #节点标签1
        label_high = x_names[sel_x] +'>%s' %break_point[sel_x] #节点标签2
        if np.unique(new_y_low).shape[0]<2 or np.unique(new_y_high).shape[0]<2:
            return y_res 
        tree[label_low] = self.cart_tree_fit(new_x_low,new_y_low,names_new,thre_num,max_depth=max_depth,now_depth=now_depth+1) #子节点递归1
        tree[label_high] = self.cart_tree_fit(new_x_high,new_y_high,names_new,thre_num,max_depth=max_depth,now_depth=now_depth+1) #子节点递归2
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
        
##GBDT尝试
class GBDT(cart_regression):
    cart_list = []
    def sigmoid(self,x):
        return 1./(1+np.exp(-x))
    def cart_tree_fit(self,x,y,x_names,thre_num=5,
                      max_depth=3,now_depth=0): #重写CART树的训练函数
        tree = {}
        y_res = 1/2. * float(sum(y))/(sum(np.abs(y)*(1-np.abs(y))))
        if y.shape[0] <= thre_num: #停止条件：当Y的数量低于阈值
            return y_res
        if now_depth >= max_depth:
            return y_res
        sel_x = 0
        new_cut_x, break_point = self.new_cut_x(x,y) #提取各个X字段切分点和分享以后的X
        gini_min = self.res_calc(x[:,0],y)
        for j in range(1,new_cut_x.shape[1]): #找出误差最小的字段
            if self.res_calc(new_cut_x[:,j],y) < gini_min:
                gini_min = self.res_calc(new_cut_x[:,j],y)
                sel_x = j  
        new_x_low = x[x[:,sel_x]<=break_point[sel_x],:] 
        new_x_high = x[x[:,sel_x]>break_point[sel_x],:]
        new_y_low = y[x[:,sel_x]<=break_point[sel_x]] #当前字段低于切分点的Y数据集
        new_y_high = y[x[:,sel_x]>break_point[sel_x]] #当前字段高于切分点的Y数据集
        names_new = x_names#[np.array(range(x_names.shape[0]))!=sel_x]
        label_low = x_names[sel_x] +'<=%s' %break_point[sel_x] #节点标签1
        label_high = x_names[sel_x] +'>%s' %break_point[sel_x] #节点标签2
        if np.unique(new_y_low).shape[0]<2 or np.unique(new_y_high).shape[0]<2:
            return y_res 
        tree[label_low] = self.cart_tree_fit(new_x_low,new_y_low,names_new,thre_num,max_depth=max_depth,now_depth=now_depth+1) #子节点递归1
        tree[label_high] = self.cart_tree_fit(new_x_high,new_y_high,names_new,thre_num,max_depth=max_depth,now_depth=now_depth+1) #子节点递归2
        if tree[label_low] == tree[label_high]:
            return tree[label_high]
        return tree
    def cart_predict_line(self,x,x_names,model): #重写CART树的单行预测函数
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
    def cart_predict(self,x,x_names,model): #重写CART树预测函数
        if model == {}:
            return "Please fit the model"
        else:
            result = []
            for i in range(x.shape[0]):
                result.append(self.cart_predict_line(x[i,:],x_names,model))
            result = np.array(result)
            return result
    def init_value(self,y):
        return np.zeros(y.shape[0])
    
    def gbdt_predict(self,x,x_names,n_tree=-1,step=0.1):
        y_init = self.init_value(x)
        if self.cart_list == []:
            return self.sigmoid(y_init)        
        elif n_tree <= 0:
            for i in range(len(self.cart_list)):
                pre_tmp = self.cart_predict(x,x_names,self.cart_list[i])
                y_init += pre_tmp * step
            return self.sigmoid(y_init)
        else:
            for i in range(n_tree):
                pre_tmp = self.cart_predict(x,x_names,self.cart_list[i])
                y_init += pre_tmp * step
            return self.sigmoid(y_init)
    def gbdt_predict_type(self,x,x_names,n_tree=-1,thre_value=0.5,step=0.1):
        y_pre = self.gbdt_predict(x,x_names,n_tree,step=step)
        for i in range(y_pre.shape[0]):
            if y_pre[i] > thre_value:
                y_pre[i] = 1
            else:
                y_pre[i] = 0 
        return y_pre
    def GBDT_fit(self,x,y,x_names,thre_num=10,
                      max_depth=5,n_tree=3,step=0.1):
        self.cart_list = []
        y_pre = self.gbdt_predict(x,x_names,step=step)
        for i in range(n_tree):
            y_gd = y - y_pre
            model_tmp = self.cart_tree_fit(x,y_gd,x_names,thre_num=thre_num,
                                           max_depth=max_depth)
            self.cart_list.append(model_tmp)
            y_pre = self.gbdt_predict(x,x_names,n_tree=-1)

#horse数据集测试
gbdt_model=GBDT() #实例化
gbdt_model.GBDT_fit(horse_train_x,horse_train_y,horse_name,thre_num=10,max_depth=2,n_tree=100) #模型训练    
for j in [1,3,5,8,10,15,20,25,30,50,75,100]: #对比采用不同的数量的子树模型的预测的效果
    gbdt_model_pre = gbdt_model.gbdt_predict_type(horse_test_x,horse_name,n_tree=j) #预测
    gbdt_model_pre_train = gbdt_model.gbdt_predict_type(horse_train_x,horse_name,n_tree=j) #预测
    ACC = sum(gbdt_model_pre == horse_test_y)/float(len(horse_test_y))
    ACC_train = sum(gbdt_model_pre_train == horse_train_y)/float(len(horse_train_y))
    print "the depth: %s, the n_tree: %s, the Accuracy to train: %s, the Accuracy to test: %s" % (i,j,ACC_train,ACC)


#利用sklearn进行GBDT建模
from sklearn.ensemble import GradientBoostingClassifier #加载模块
#实例化模型，传入参数（学习率：0.1，最小叶节点样本数：3，最小划分叶节点样本数：20，最大树深：7，子采样比例：0.8，迭代次数：40
GBDT_model = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=3,
min_samples_leaf=20,max_depth=7, subsample=1,n_estimators=40)
GBDT_model.fit(horse_train_x,horse_train_y) #训练模型
y_pre_GBDT=GBDT_model.predict(horse_test_x) #预测模型
sum(y_pre_GBDT==horse_test_y)/float(horse_test_y.shape[0]) #准确度，当前GBDT的模型对于horse测试集的预测准确度超过了前述所有模型

from sklearn.grid_search import GridSearchCV #网格搜索模块

clf = GradientBoostingClassifier()
#传入建模参数learning_rate和n_estimators、max_depth，候选值的范围包含了整数3-20
parameters = {'learning_rate':np.array([0.1,0.2,0.3,0.4,0.5]),'n_estimators':np.array([25,50,100]),'max_depth':np.array([2,3,4,5,6,7])}
#网格参数搜索，输入之前的模型流程pipe_process,候选参数parameters，并且设置5折交叉验证
gs_GBDT = GridSearchCV(clf,parameters,verbose=2,refit=True,cv=5) #设置备选参数组
gs_GBDT.fit(horse_train_x,horse_train_y) #模型训练过程
print gs_GBDT.best_params_,gs_GBDT.best_score_ #查看最佳参数和评分（准确度）
#最佳参数的KNN建模对预测数据预测效果）
print 'The Accuracy of GradientBoostingClassifier model with best parameter and MinMaxScaler is',gs_GBDT.score(horse_test_x,horse_test_y)

#XGBoost
import xgboost as xgb
xgb_model = xgb.XGBClassifier(max_depth=2,n_estimators=100)
xgb_model.fit(horse_train_x,horse_train_y)
y_pre_xgb = xgb_model.predict(horse_test_x)
sum(y_pre_xgb==horse_test_y)/float(horse_test_y.shape[0]) #准确度，当前GBDT的模型对于horse测试集的预测准确度超过了前述所有模型






