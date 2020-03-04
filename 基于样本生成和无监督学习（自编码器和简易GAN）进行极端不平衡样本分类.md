# 样本极端不平衡时可以参考的方法
&#8195;&#8195;
Sample:以KAGGLE信用卡反欺诈数据，含标签和10个字段，通过抽样造出样本极度不平衡的情况（badrate = 0.5%）
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression #加载模块
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pd.set_option('display.max_columns', 30)

#KS指标计算函数
def evalKS(preds, Y):       # written by myself
    fpr, tpr, thresholds = roc_curve(Y, preds)
    #print 'KS_value', (tpr-fpr).max()
    return 'KS_value', np.abs(tpr-fpr).max()

#读取数据
credit_data = pd.read_csv('cs-training.csv')
del(credit_data['Unnamed: 0'])

#抽样使得研究的样本的阳性比例约0.5%，符合极端不平衡样本的要求
black_spl = credit_data.loc[credit_data.SeriousDlqin2yrs==1].sample(200,random_state=100)
white_spl = credit_data.loc[credit_data.SeriousDlqin2yrs==0].sample(39800,random_state=100)
credit_data2 = black_spl.append(white_spl).reset_index(drop=True) 

#划分训练集合测试集，以测试集来评估效果
x_var = [v for v in  credit_data2.columns if v != 'SeriousDlqin2yrs']
train_X, test_X, train_Y, test_Y = train_test_split(credit_data2[x_var], credit_data2['SeriousDlqin2yrs'],test_size=0.4, random_state=50)

#对缺失值以0进行简单填补
train_X = train_X.fillna(0)
test_X = test_X.fillna(0)

#当前样本简单训练Logistic模型，作为参照
lr_model = LogisticRegression(C=10, random_state=0) #实例化
lr_model.fit(train_X,train_Y) #模型训练

#评估KS和AUC效果
print('Trian KS Logistic: ',evalKS(lr_model.predict_proba(train_X)[:,1], train_Y))
print('Trian AUC Logistic: ',roc_auc_score(train_Y,lr_model.predict_proba(train_X)[:,1]))

print('Test KS Logistic: ',evalKS(lr_model.predict_proba(test_X)[:,1], test_Y))
print('Test AUC Logistic: ',roc_auc_score(test_Y,lr_model.predict_proba(test_X)[:,1]))
```
## 过采样Oversampling
```python
#############################过采样###############################
#过采样函数定义
def over_sampling(data, #X
                  label, #Y
                  N_sample = 10000, #过采样以后的样本数量
                  rate_1 = 0.1, #新的样本中阳性占比
                  random_state = 100):
    num_1 = int(N_sample * rate_1)
    num_0 = N_sample - num_1
    tmp_1 = data.loc[label==1].sample(num_1, random_state=random_state)
    tmp_0 = data.loc[label==0].sample(num_0, random_state=random_state*2)
    sample_res = tmp_1.append(tmp_0).reset_index(drop=True)
    label_res = pd.Series([1]*num_1 + [0]*num_0)
    return(sample_res,label_res)

#过采样样本
trian_X_oversmp, trian_Y_oversmp = over_sampling(train_X,
                                                 train_Y,
                                                 10000,
                                                 0.0121)

#基于过采样样本训练的LR模型
lr_model_oversmp = LogisticRegression(C=10, random_state=0) #实例化
lr_model_oversmp.fit(trian_X_oversmp,trian_Y_oversmp) #模型训练

#模型效果评估
print('Trian KS Oversmp_Logistic: ',evalKS(lr_model_oversmp.predict_proba(train_X)[:,1], train_Y))
print('Trian AUC Oversmp_Logistic: ',roc_auc_score(train_Y,lr_model_oversmp.predict_proba(train_X)[:,1]))

print('Test KS Oversmp_Logistic: ',evalKS(lr_model_oversmp.predict_proba(test_X)[:,1], test_Y))
print('Test AUC Oversmp_Logistic: ',roc_auc_score(test_Y,lr_model_oversmp.predict_proba(test_X)[:,1]))
```

## SMOTE算法生成新样本
基于K近邻的SMOTE方法可以生成新样本，调和两类样本的比例。
```python
#########################smote算法生成样本#######################
'''
smote简介
1.对N个少数类样本，选定其中n个样本
2.对n个样本中每个样本进行K近邻聚类，每个样本找出其最近的k个少数类样本
3.对k个样本中选择m个样本和当前样本的连线，在每条连线上随机寻找一个点，即为m个新的样本
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#抽取坏样本并且标准化样本剔除量纲影响
scale_tsf_B = StandardScaler()
Black_data_B = scale_tsf_B.fit_transform(train_X.loc[train_Y==1])

#定义SMOTE样本生成方法
def new_sample_create(data, 
                      N=50, #生成新样本的旧样本点数
                      K=5, #K近邻数 
                      m=2, #每个样本点和m个近邻产生m个新样本点（共N*m个新样本）
                      seed=100):
    #训练K近邻模型
    KN_model = KNeighborsClassifier(n_neighbors = K+1)
    y_tmp = np.array([1]*data.shape[0])
    KN_model.fit(data,y_tmp)
    #抽取用于近邻算法的预测点
    np.random.seed(seed)
    N_idx = np.random.choice(np.array(range(data.shape[0])),N,replace=False).tolist()
    data_smp = data[N_idx,:]
    #提取每个预测点的K近邻点的idx
    kn_smp = KN_model.kneighbors(data_smp,return_distance=False)[:,1:]
    #对N个预测点，将K个近邻中找出最近的m个点与之直线距离之间的随机一点作为生成的新样本
    sample_create = np.array([0]*data.shape[1])
    for i in range(N):
        np.random.seed(seed + i*10)
        random_factor = np.random.uniform(0,1,m).reshape(m,1)
        smp_tmp = data_smp[i,:] + random_factor * (data[kn_smp[i,:m],:] - data_smp[i,:])
        sample_create = np.vstack((sample_create, smp_tmp))
    return(sample_create[1:,:])

#SMOTE生成新的坏样本（360个）
N_smp_use = 120
m = 3
Black_sample_new = new_sample_create(Black_data_B, N=N_smp_use, K=5, m=m, seed=200)
Black_sample_new = scale_tsf_B.inverse_transform(Black_sample_new)
#将生成的新样本追加到训练集中，此时阳性样本比例提高了近3倍（亦可以考虑适当降低阴性样本比例进一步平衡）
train_X_new = np.vstack((train_X,Black_sample_new))
train_Y_new = train_Y.append(pd.Series(np.array([1]*(N_smp_use*m))))

#基于SMOTE样本的结果
lr_model_smote = LogisticRegression(C=10, random_state=0) #实例化
lr_model_smote.fit(train_X_new,train_Y_new) #模型训练

print('Trian KS Smote_Logistic: ',evalKS(lr_model_smote.predict_proba(train_X)[:,1], train_Y))
print('Trian AUC Smote_Logistic: ',roc_auc_score(train_Y,lr_model_smote.predict_proba(train_X)[:,1]))

print('Test KS Smote_Logistic: ',evalKS(lr_model_smote.predict_proba(test_X)[:,1], test_Y))
print('Test AUC Smote_Logistic: ',roc_auc_score(test_Y,lr_model_smote.predict_proba(test_X)[:,1]))
```

## AutoEncoder自编码器异常点检测
```python
######################自编码器识别####################
'''
AutoEncoder将数据输入经过编码--解码，最终输出接近输入的数据
将占比绝大部分的非欺诈样本视为正常样本，问题转变为对异常点（欺诈样本）的识别 
通过将正常样本输入具有一定层数AutoEncoder进行编码--解码操作
训练以后的AutoEncoder模型应该是对正常样本输出的结果与输入的正常数据的误差应该相对较小，
对于没有参与训练的少数欺诈样本，经过AutoEncoder模型的输出应该会和输入数据存在较大的误差
'''
from keras import Model
from keras.layers import Dense, Input, Dropout

#定义自编码器模型
#定义编码器
encoder_input = Input(shape=(10,))
encoder = Dense(16,activation='relu')(encoder_input)
encoder = Dense(32,activation='relu')(encoder)
#定义解码器
decoder = Dense(32,activation='relu')(encoder)
decoder = Dense(16,activation='relu')(decoder)
decoder = Dropout(0.3)(decoder) #Dropout30%的参数避免过拟合
decoder = Dense(10,activation='linear')(decoder) #输出函数：线性函数
#定义最终模型
auto_encoder_model = Model(encoder_input,outputs = decoder)
encoder_model = Model(encoder_input,outputs = encoder)
#自编码器模型编译
auto_encoder_model.compile(optimizer='Adam', loss='mse')

#标准化以消除量纲，参与模型训练的只有白样本
scale_tsf_W = StandardScaler()
White_data_W = scale_tsf_W.fit_transform(train_X.loc[train_Y==0])
Black_data_W = scale_tsf_W.transform(train_X.loc[train_Y==1])
train_X_std_W = scale_tsf_W.transform(train_X)

#模型训练
auto_encoder_model.fit(x=White_data_W,
                       y=White_data_W,
                       epochs=50, 
                       batch_size=128,
                       shuffle=False)
#自编码器在黑样本和白样本的输入--输出误差
#error_edis_Black = np.sqrt(np.square((Black_data_W) - auto_encoder_model.predict(Black_data_W)).sum(1))
#error_edis_White = np.sqrt(np.square((White_data_W) - auto_encoder_model.predict(White_data_W)).sum(1))

#计算所有原始样本经过AutoEncoder的输入和输出的欧氏距离（即误差）
error_edis = np.sqrt(np.square((train_X_std_W) - auto_encoder_model.predict(train_X_std_W)).sum(1))
error_edis_test = np.sqrt(np.square((scale_tsf_B.transform(test_X)) - auto_encoder_model.predict(scale_tsf_B.transform(test_X))).sum(1))

#结果评估
print('Train KS autoencoder: ',evalKS(1.0/error_edis, train_Y))
print('Train AUC autoencoder: ',roc_auc_score(train_Y,error_edis))

print('Test KS autoencoder: ',evalKS(1.0/error_edis_test,test_Y))
print('Test AUC autoencoder: ',roc_auc_score(test_Y,error_edis_test))

```

## 生成对抗神经网络GAN

```python
###################简易GAN实现########################
'''
GAN通过训练两个具有若干层的神经网络：生成器和判别器，前者生成样本、后者判别样本
在训练通过两个模型相互‘博弈’，使得生成器生成更为接近真实的样本，而判别器也会具备更高的对生成样本和真实样本的区分度
本例中利用白样本进行训练，得到的判别器模型对真实的白样本和黑样本有较高的识别能力
'''

from keras import Sequential, Model
from keras.layers import Dense, LeakyReLU,Flatten, BatchNormalization,Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt

class simple_GAN(object):
    def __init__(self,
                 data, #数据输入
                 G_input_shape, #生成器输入维度
                 G_output_shape, #生成器输出维度（即判别器输入维度）
                 optimizer = Adam(0.0002, 0.5), #估计器
                 batch_size = 100, #单次迭代处理样本数量
                 epochs = 1000): #迭代次数
        self.data = data
        self.G_input_shape = G_input_shape
        self.G_output_shape = G_output_shape
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
    
    #生成器定义函数
    #结构 输入--32--64--128--输出
    def define_generator(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.G_input_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.G_output_shape, activation='sigmoid'))
        #model.summary()
        noise = Input(shape=(self.G_input_shape,))
        img = model(noise)
        self.Generator = Model(noise, img)
    
    #判别器定义函数
    #结构 输入--64--32--输出真伪
    def build_discriminator(self):
    
        model = Sequential()
        model.add(Dense(64,input_dim=self.G_output_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        #model.summary()
        ipt = Input(shape=(self.G_output_shape,))
        validity = model(ipt)
        self.Discriminator = Model(ipt, validity)
    
    #模型编译
    def model_compile(self,):
        #判别器编译
        self.Discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        z = Input(shape = (self.G_input_shape, ))
        fake_smp = self.Generator(z) 
        validity = self.Discriminator(fake_smp)
        self.Combind = Model(z, validity)
        #在生成器和判别器融合模型中只涉及生成器的参数更新，故冻结融合模型中的判别器参数
        self.Discriminator.trainable = False
        #融合模型（迭代生成器）编译
        self.Combind.compile(loss='binary_crossentropy',
                             optimizer=optimizer)
    #模型训练
    def model_train(self,):
        #真假标签
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))
        
        for epoch in range(self.epochs):
            idx = np.random.randint(0, self.data.shape[0], self.batch_size)
            sample = self.data[idx,:]
        
            noise = np.random.normal(0, 1, (self.batch_size, self.G_input_shape))
            gen_smp = self.Generator.predict(noise)
            d_loss_real = self.Discriminator.train_on_batch(sample, valid)
            d_loss_fake = self.Discriminator.train_on_batch(gen_smp, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            noise = np.random.normal(0, 1, (self.batch_size, self.G_input_shape))
            g_loss = self.Combind.train_on_batch(noise, valid)
            #输出误差、ACC信息
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

#01标准化
scale_tsf_W = MinMaxScaler()
#scale_tsf_W = StandardScaler()
White_data_W = scale_tsf_W.fit_transform(train_X.loc[train_Y==0])
Black_data_W = scale_tsf_W.transform(train_X.loc[train_Y==1])
train_X_std_W = scale_tsf_W.transform(train_X)
test_X_std_W = scale_tsf_W.transform(test_X)

#模型定义和训练
White_Gen_model = simple_GAN(White_data_W,
                             100,
                             10,
                             optimizer = Adam(0.0002, 0.5),
                             batch_size = 128,
                             epochs = 500)  
White_Gen_model.build_discriminator()
White_Gen_model.define_generator()
White_Gen_model.model_compile()
White_Gen_model.model_train()

#评估结果
print('Train KS GAN: ',evalKS(1-White_Gen_model.Discriminator.predict(train_X_std_W),train_Y))
print('Train AUC GAN: ',roc_auc_score(1-train_Y,White_Gen_model.Discriminator.predict(train_X_std_W)))

print('Test KS GAN: ',evalKS(1-White_Gen_model.Discriminator.predict(test_X_std_W),test_Y))
print('Test AUC GAN: ',roc_auc_score(test_Y,1-White_Gen_model.Discriminator.predict(test_X_std_W)))


```



