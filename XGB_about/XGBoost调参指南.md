# XGBoost调参指南
## XGBoost 模块基本用法
### params参数  
objective（模型目标）:binary:logistic二分类，multi:softmax多分类，reg:linear线性回归  
eta（学习率，步长）：通常0.01-0.2  
max_depth（每棵树的最大深度）：通常3-10  
max_leaf_nodes（单棵树的最大节点数）：与max_depth只有一个发挥作用  
gamma（最小损失函数下降值）：只有当该节点分裂后损失函数值降低大于这个值才会分裂  
subsample（每棵树随机采样的比例）：通常0.5-1  
colsample_bytree（每棵树随机采样列数占比）：（通常0.5-1）
alpha（L1正则化参数）：  
lambda（L2参数）：  
eval_metric(目标函数):可选rmse，mae，logloss，error二分类错误率，merror，mlogloss，auc ，也可以自定义  
seed（随机种子）:  
n_jobs(并行进程数):
silent：运算时是否打印本次迭代效果
### fit参数
params(params参数):  
dtrain(xgb.DMatrix格式的数据):  
num_boost_round(迭代次数)  
evals(观察效果的数据集，验证集，列表包含元祖形式)  
feval：可以自定义的函数  
maximize：自定义的函数是否追求最大值  
early_stopping_rounds:多少次迭代没有提升则停止
### 示例
```python
import xgboost as xgb 
xgb_params = {"objective": "binary:logistic",
              "eta": 0.1, 
              "max_depth":5,
              'alpha':0.2,
              "seed": 1000, 
              "silent":0, 
              'n_jobs':12,
              'eval_metric':'auc'
}
dtrain = xgb.DMatrix(train_x,label=train_y)
dtest = xgb.DMatrix(test_x,label=test_y)
watchlist = [ (dtrain,'train'), (dtest, 'test') ]
xgb_model = xgb.train(params=xgb_params,
                      dtrain=dtrain,
                      num_boost_round=200,
                      evals=watchlist,
                      #feval=evalKS,
                      #maximize=True,
                      early_stopping_rounds=30)
xgb_model.get_fscore() #输出变量重要性结果
xgb_model.predict(dtest) #预测
```
## 交叉验证操作
参数基本类似xgb.fit  
额外有nfold参数，K折交叉验证，不需要evals
```python
xgb_cv = xgb.cv(xgb_params,
                dtrain,
                num_boost_round=500,
                nfold=5,
                feval=evalKS,
                maximize=True,
                verbose_eval=True,
                early_stopping_rounds=30)
print 'best_iter: ',xgb_cv.shape[0] #交叉验证最佳迭代次数
```
## 适合sklearn的XGBClassifier接口
XGBClassifier的训练类似于sklearn的算法模型，部分参数和xgb.train有所不同：  
eta -> learning_rate，学习步长   
lambda -> reg_lambda，L2正则系数  
alpha -> reg_alpha，L1正则系数  
```python
from xgboost.sklearn import XGBClassifier
xgb_tmp = XGBClassifier(objective="binary:logistic", 
                        learning_rate=0.1,     max_depth=5,
                        n_estimators=300,
                        reg_alpha=0.2,
                        seed=1000,
                        silent=0,
                        nthread=12)
xgb_tmp.fit(train_x,
            train_y,
            eval_set=[(train_x,train_y),(test_x, test_y)],
            eval_metric='logloss',
            early_stopping_rounds=30)
```

## 自定义评价函数
### xgb.train的定义方法
在xgb.train中的feval参数支持传入自定义的参数（此时params的eval_metric不再定义），定义的函数形式应该是func(preds,dtrain),其中dtrain是包含label的xgb.DMatrix对象（通过dtrain.get_label()获取实际标签），输出可以包含一个字符串+计算结果，其中字符串可以在每一步作为说明被打印出来。  
maximize参数定义是否使得自定义函数最大化，否则最小化
```python
from sklearn.metrics import roc_curve
#定义KS值计算函数
def evalKS(preds, dtrain):       # written by myself
    fpr, tpr, thresholds = roc_curve(dtrain.get_label(), preds)
    return 'KS_value', (tpr-fpr).max()
xgb_model = xgb.train(params=xgb_params,
                      dtrain=dtrain,
                      num_boost_round=200,
                      evals=watchlist,
                      feval=evalKS,
                      maximize=True,
                      early_stopping_rounds=30)
```
### XGBClassifier的定义方法
在XGBClassifier的fit中的eval_metric可以传入自定义函数，但是只能做到使得该函最小化（如KS这种指标需要将符号反过来以达成目标）
```python
def evalKS2(preds, dtrain):       # written by myself
    fpr, tpr, thresholds = roc_curve(dtrain.get_label(), preds)
    return 'KS_value', -(tpr-fpr).max()
xgb_tmp.fit(train_x,
            train_y,
            eval_set=[(train_x,train_y),(test_x, test_y)],
            eval_metric=evalKS2,
            early_stopping_rounds=30)
```
## GridSearchCV网格调参
XGBClassifier的方法支持sklearn的GridSearchCV的网格调参方法。


```python
#自定义scoring参数
def evalKS3(model, X, y):       # written by myself
    #print 'now start ks calc'
    fpr, tpr, thresholds = roc_curve(y, model.predict_proba(X)[:,1])
    #print 'now end ks calc'
    return (tpr-fpr).max()

param_test1 = {
 'max_depth':range(3,10,2),
 'learning_rate':[0.05,0.075,0.1,0.15,0.2],
 'reg_alpha':[0.05,0.1,0.2,0.3],
}
fit_params1={
 'eval_set':[(train_x,train_y),(test_x,test_y)],
 'eval_metric':evalKS2,
 'early_stopping_rounds':30
        }

gsearch1 = GridSearchCV(estimator = XGBClassifier(n_estimators=300,
                                                  min_child_weight=1, 
                                                  gamma=0, 
                                                  subsample=0.8,
                                                  colsample_bytree=0.8,
                                                  objective= 'binary:logistic', 
                                                  nthread=6,
                                                  scale_pos_weight=1,
                                                  silent=1, 
                                                  seed=1000), 
                        param_grid = param_test1, 
                        scoring=evalKS3,
                        fit_params = fit_params1, 
                        n_jobs=6, 
                        iid=True, 
                        cv=5)
gsearch1.fit(train_x,train_y)

gsearch1.grid_scores_  #各个参数组合对应的分数
gsearch1.best_params_  #最佳参数组合 
gsearch1.best_score_   #最高评分

```

## 附：Lightgbm调参
```python
from sklearn.datasets import load_breast_cancer
import lightgbm as lgb
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV 
from sklearn.metrics import accuracy_score
from lightgbm.sklearn import LGBMClassifier

data_X = load_breast_cancer()['data']
data_Y = load_breast_cancer()['target']

train_X,test_X,train_Y,test_Y = train_test_split(data_X,data_Y,test_size=0.2, random_state=66)

lgb0 = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_error',
        eval_set = [(train_X,train_Y),(test_X,test_Y)], #交叉验证对照
        eval_names = ['train','test'],
        num_leaves=31, #树模型复杂度（叶节点总数上限）
        max_depth=4, #树深
        learning_rate=0.05, #学习率
        seed=2018, #随机种子
        colsample_bytree=0.8,
        min_child_samples=8,
        subsample=0.9,
        n_estimators=10)

lgb0.fit(train_X,train_Y,eval_set=[(train_X,train_Y),(test_X,test_Y)])

accuracy_score(test_Y,lgb0.predict(test_X))
accuracy_score(train_Y,lgb0.predict(train_X))


param_test1 = { #备选模型入参
 'max_depth':[2,3,4], #最大树深
 'learning_rate':[0.05,0.1], #学习率（步长）
 'reg_alpha':[0.05,0.1]
}
fit_params1={ #训练参数设置
        'eval_set':[(train_X,train_Y),(test_X,test_Y)],
        'eval_metric':"logloss",
        'early_stopping_rounds':5
        }


gsearch1 = GridSearchCV(estimator = LGBMClassifier(n_estimators=10,
                                                   #num_leaves = 12, #叶节点数
                                                   objective='binary', 
                                                   n_jobs=2,
                                                   seed=1000), 
                        param_grid = param_test1, 
                        scoring='accuracy',
                        fit_params = fit_params1, 
                        n_jobs=2, 
                        iid=True, 
                        cv=5)
gsearch1.fit(train_X,train_Y)

gsearch1.grid_scores_
gsearch1.best_params_
gsearch1.best_score_

````













