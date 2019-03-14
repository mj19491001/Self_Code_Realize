# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:40:18 2017

@author: JQstyle
"""
#加载numpy模块
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


#梯度上升求解LR
#L= (Yln(1/(1+exp(-XK)))+(1-Y)*ln(1-1/(1+exp(-XK)))) ==> d(L)/d(K) = X.T•(Y-1/(1+exp(-XK)))
#K_n+1 = K_n + d(L)/d(K_n) * s
class Logistic_Regression():
    def log_fun(self,x): #sigmoid函数
        return 1/(1+np.exp(-x))
        
    def fit_gd(self,x,y,iter=1000,s=0.01,e_break=0.001): #逻辑回归参数求解
        x = np.matrix(np.hstack((np.ones((x.shape[0],1)),x)))
        length = x.shape[1]
        K_init = np.matrix(np.random.rand(length)-0.5).T #初始化系数
        y = np.matrix(y).T
        i = 1
        while i <= iter:#梯度上升迭代
            g = x.T.dot(np.array((y - self.log_fun(x.dot(K_init)))))
            K_init = K_init + g*s
            i += 1
        self.a = K_init
        return self.a
        
    def predict_prob(self,x): #预测函数（概率）
        return np.array(self.log_fun(x.dot(self.a[1:,:])+self.a[0,0]).T)[0,:]

    def predict_type(self,x,thre_var=0.5): #预测函数（类别）
        type_pre = self.predict_prob(x)
        for i in range(type_pre.shape[0]):
            if type_pre[i] > thre_var:
                type_pre[i] = 1
            else:
                type_pre[i] = 0
        return type_pre
        
    def predict_accuracy(self,x,y,thre_var=0.5): #预测准确度函数
        y_pre = self.predict_type(x,thre_var)
        accuracy = sum(y_pre==y)/float(y.shape[0])
        return accuracy
        
    def fit_gd_L2(self,x,y,iter=1000,s=0.01,l2=0.1): #逻辑回归参数求解(L2)
        x = np.matrix(np.hstack((np.ones((x.shape[0],1)),x)))
        length = x.shape[1]
        K_init = np.matrix(np.random.rand(length)-0.5).T #初始化参数
        y = np.matrix(y).T
        i = 1
        while i <= iter:#梯度下降迭代(添加了L2正则项)    
            K_init = K_init + x.T.dot(np.array(y - self.log_fun(x.dot(K_init))))*s - l2*K_init*s
            i += 1
        self.a = K_init
        return self.a


logistic_model = Logistic_Regression() #实例化模型对象
logistic_model.fit_gd(horse_train_x,horse_train_y,iter=10000,s=0.001) #模型训练
horse_pred = logistic_model.predict_type(horse_test_x) #预测
logistic_model.predict_accuracy(horse_test_x,horse_test_y) #准确度

#采用sklearn模块的逻辑回归示例
from sklearn.linear_model import LogisticRegression #加载模块
lr = LogisticRegression(C=1000, random_state=0) #实例化
lr.fit(horse_train_x,horse_train_y) #模型训练
sum(lr.predict(horse_train_x) == horse_train_y)/float(len(horse_train_y)) #分类准确度


#附：拟牛顿法求解LogisticRegression
#相比于梯度下降对模型的求解（一阶收敛），牛顿法，拟牛顿法具有更快的下降收敛速度（二阶收敛）
#牛顿法是利用目标函数的二阶泰勒展开式作为其替代函数，
#并且得到对应的梯度，按照极值点梯度为0的思想，求出极值点xi
#用新的xi会带入二阶泰勒展开式的梯度中，重复迭代，直到停止

#因为牛顿法涉及二阶展开的海塞矩阵及其求逆运算，该步骤较为复杂
#通过对海塞矩阵H的逆矩阵的估计来改进和简化的方法为拟牛顿法
#DFP作为常见的拟牛顿法之一的算法，采用一个满足条件的正定矩阵G来替代H的逆
#每次迭代通过两个附加项P和Q来更新G

class logistic_NTmodel():
    def log_fun(self,x): #sigmoid函数
        return 1/(1+np.exp(-x))

    def pos_def_ma(self,n): #随机产生一个正定矩阵（用于拟牛顿法）
        import numpy as np
        a = np.diag(np.random.rand(n))
        c = np.array(np.random.rand(n*n)).reshape(n,n)       
        a = c.T.dot(a).dot(c)
        return(a)
    
    def grad_los(self,w,x,y): #logistic损失函数的一阶导数（梯度）
        grad = -x.T.dot(np.array(y - self.log_fun(x.dot(w))))
        return(grad)

    def hessi_mat(self,w,x): #海塞矩阵函数
        hessi_mat = np.zeros([x.shape[1],x.shape[1]])
        x_array=np.array(x)
        for i in range(x.shape[1]):
            for j in range(x.shape[1]):
                hessi_mat[i,j] = (np.matrix(x_array[:,i]*x_array[:,j]).dot(np.array(
                        self.log_fun(x.dot(w)))*np.array(1-self.log_fun(x.dot(w)))))[0,0]
        return(np.matrix(hessi_mat))

    def fit_NT(self,x,y,iter=100): #牛顿法拟合函数
        import numpy as np   
        x = np.matrix(np.hstack((np.ones((x.shape[0],1)),x)))
        y = np.matrix(y).T
        length = x.shape[1]
        #K_init = np.matrix(np.random.rand(length)-0.5).T #初始化参数
        K_init = np.matrix(np.zeros(length)).T #初始化参数（0向量）
        i = 1
        while i <= iter:
            gi = self.grad_los(K_init,x,y) #产生梯度
            Hi = self.hessi_mat(K_init,x) #产生海塞举证
            #print Hi
            pi = -Hi.I.dot(gi) #计算p
            K_init = K_init + pi #更新模型系数
            i += 1
        self.a = K_init
        return(self.a)
    
    def fit_DFP(self,x,y,iter=100,a = 0.01): #DFP拟牛顿算法函数(p的系数取固定值a)
        import numpy as np   
        x = np.matrix(np.hstack((np.ones((x.shape[0],1)),x)))
        y = np.matrix(y).T
        length = x.shape[1]
        G = self.pos_def_ma(x.shape[1])  #初始正定矩阵G（用以替代牛顿法海塞矩阵（逆矩阵））
        #K_init = np.matrix(np.random.rand(length)-0.5).T #初始化参数
        K_init = np.matrix(np.zeros(length)).T
        i = 1
        gi = self.grad_los(K_init,x,y) #梯度计算
        while i <= iter:        
            pi = -G.dot(gi)
            K_init_old = K_init.copy()
            K_init = K_init + a*pi
            gi_old = gi.copy() 
            gi = self.grad_los(K_init,x,y) #梯度更新计算
            gi_dif = gi - gi_old
            K_init_dif = K_init - K_init_old
            Pi = K_init_dif.dot(K_init_dif.T)/(K_init_dif.T.dot(gi_dif)) #正定矩阵G附加项1
            Qi = -G.dot(gi_dif).dot(gi_dif.T).dot(G)/(gi_dif.T.dot(G).dot(gi_dif)) #正定矩阵G附加项2
            G = G + Pi + Qi #更新正定矩阵G
            i += 1
        self.a = K_init
        return(self.a)
        
    def predict_prob(self,x): #预测函数（概率）
        return np.array(self.log_fun(x.dot(self.a[1:,:])+self.a[0,0]).T)[0,:]

    def predict_type(self,x,thre_var=0.5): #预测函数（类别）
        type_pre = self.predict_prob(x)
        for i in range(type_pre.shape[0]):
            if type_pre[i] > thre_var:
                type_pre[i] = 1
            else:
                type_pre[i] = 0
        return type_pre
        
    def predict_accuracy(self,x,y,thre_var=0.5): #预测准确度函数
        y_pre = self.predict_type(x,thre_var)
        accuracy = sum(y_pre==y)/float(y.shape[0])
        return accuracy

#牛顿法求解LOGISTIC示例：
model_lr_NT = logistic_NTmodel() #实例化模型
model_lr_NT.fit_NT(horse_train_x,horse_train_y,iter=15) #牛顿法训练模型
model_lr_NT.predict_accuracy(horse_test_x,horse_test_y) #预测准确度

#拟牛顿法求解LOGISTIC示例
model_lr_DFP = logistic_NTmodel() #实例化模型
model_lr_DFP.fit_DFP(horse_train_x,horse_train_y,iter=2000,a=0.0001) #拟牛顿法训练模型     
model_lr_DFP.predict_accuracy(horse_test_x,horse_test_y) #预测准确度

#附：对逻辑回归正则化的考虑
#对逻辑回归正则化的考虑是为了尽量避免过拟合的情况
#在此介绍L2正则项
#考虑到正则化的损失函数：los = -(Yln(1/(1+exp(-XK)))+(1-Y)*ln(1-1/(1+exp(-XK)))) + L2
#其中， L2= l*K.T•K
#此时的梯度： d(los)/d(K) = -X.T•(Y-1/(1+exp(-XK))) + l*K
#l的大小决定了对过拟合的重视程度
#考虑到L2正则的逻辑回归梯度下降求解函数：

#对于L2的实现，已将函数加入到之前的梯度下降的Logistic_Regression类中









