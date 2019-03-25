#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:47:35 2019

@author: JQstyle
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score  
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

#used for bin_seq2_n
seq_dir = {'[':lambda x,a: x>=a,
           '(':lambda x,a: x>a,
           ']':lambda x,a: x<=a,
           ')':lambda x,a: x<a}

#分箱函数：较高效率   
def bin_seq2_n(X,range_df,seq_dir=seq_dir,label='range',other_fill=-999):  
    def range_in(x,r_tmp):
        try:
            return (x == r_tmp or str(x) == r_tmp or x == float(r_tmp))
        except:
            return (seq_dir[r_tmp[0]](x,r_tmp[1]) and seq_dir[r_tmp[-1]](x,r_tmp[-2]))
    X_res = pd.Series([other_fill]*X.shape[0])
    for r in range_df['range']:
        try:
            r_tmp = str(r).replace(' ','').split(',')
            r_tmp = [r_tmp[0][0]] + [float(r_tmp[0].replace(r_tmp[0][0],''))] + [float(r_tmp[1].replace(r_tmp[1][-1],''))] + [r_tmp[1][-1]]
        except:
            r_tmp = r
        X_res[X.map(lambda x: range_in(x,r_tmp)) & (X_res == other_fill)] = range_df[label][range_df['range']==r].iloc[0] 
    return X_res

#单变量分箱
def woe_single(X,
               Y,
               bad = 1,
               min_bin=0.05,
               allow_u = False
               ):
    if bad == 0:
        Y = 1 - Y
    #单一值大于20的情况
    if X.nunique() > 10:
        bad = Y.sum()  # 坏客户数(假设因变量列为1的是坏客户)
        good = Y.count() - bad  # 好客户数
        n = 10
        #while np.abs(r) < 0.7 or n > 15:
        cut = X.quantile([i*1.0/n for i in range(n+1)]).drop_duplicates().sort_values().tolist()
        cut = [cut[0]-0.01] + cut
        d1 = pd.DataFrame({"X": X, "Y": Y, "RANGE": pd.cut(X, cut, right=True, duplicates='drop')})      
        while d1.shape[1] > 2:
            d1 = pd.DataFrame({"X": X, "Y": Y, "RANGE": pd.cut(X, cut,right=True, duplicates='drop')})
            d2 = d1.groupby('RANGE', as_index=False)
            d3 = pd.DataFrame(d2.X.min(), columns=['min'])
            d3['min'] = d2.min().X
            d3['max'] = d2.max().X
            d3['num_bad'] = d2.sum().Y
            d3['total'] = d2.count().Y
            d3['bad_rate'] = d2.mean().Y
            d3['group_rate'] = d3['total'] / (bad + good)
            d3['woe'] = np.log((d3['bad_rate'] / (1 - d3['bad_rate'])) / (bad / float(good)))
            d3['iv'] = (d3['num_bad'] / bad - ((d3['total'] - d3['num_bad']) / good)) * d3['woe']
            iv = d3['iv'].sum()
            d3['iv_sum'] = iv
            cut1 = list(d3['min'].round(4))
            cut = list(d3['max'].round(4))
            cut = [cut1[0]-0.01] + cut
            cut.pop()
            cut.append(float('inf'))
            #print cut
            woe_dif = (d3['woe'][1:d3.shape[0]].reset_index(drop=True) - d3['woe'][:(d3.shape[0]-1)].reset_index(drop=True))
            #print d3
            if (woe_dif>0).sum() == d3.shape[0]-1 or (woe_dif>0).sum() == 0:
                break
            elif allow_u:
                b = 0
                for i in range(1,len(woe_dif)):
                    if woe_dif[i]*woe_dif[i-1] < 0:
                        b += 1
                if b <=1:
                    break
            woe_dif = woe_dif.map(lambda x: np.abs(x))
            d = d3['max'][:-1].reset_index(drop=True)[woe_dif == woe_dif.min()].iloc[0]
            try:
                cut.remove(d)
            except:
                for i in cut:
                    if np.abs(d - i) <= 0.0001:
                        cut.remove(i)
                        break
        #d3['range'] = d1["Bucket"].drop_duplicates().sort_values()
        d3['range'] = ['('+str(cut[i])+', '+str(cut[i+1])+']' for i in range(len(cut)-1)]
        d3['range'] = d3['range'].astype('str').str.replace('inf]','inf)')
    #单一值小于10的情况
    else:
        bad = Y.sum()  # 坏客户数
        good = Y.count() - bad  # 好客户数
        cut = X.drop_duplicates().sort_values().tolist()
        cut = [cut[0]-0.01] + cut
        d1 = pd.DataFrame({"X": X, "Y": Y, "RANGE": pd.cut(X, cut, right=True, duplicates='drop')})
        d2 = d1.groupby('RANGE', as_index=False)
        d3 = pd.DataFrame(d2.X.min(), columns=['min'])
        d3['min'] = d2.min().X
        d3['max'] = d2.max().X
        d3['num_bad'] = d2.sum().Y
        d3['total'] = d2.count().Y
        d3['bad_rate'] = d2.mean().Y
        d3['group_rate'] = d3['total'] / (bad + good)
        d3['woe'] = np.log((d3['bad_rate'] / (1 - d3['bad_rate'])) / (bad / float(good)))
        d3['iv'] = (d3['num_bad'] / bad - ((d3['total'] - d3['num_bad']) / good)) * d3['woe']
        iv = d3['iv'].sum()
        d3['iv_sum'] = iv
        while d1.shape[1] > 2:
            cut1 = list(d3['min'].round(4))
            cut = list(d3['max'].round(4))
            cut = [cut1[0]-0.01] + cut
            cut.pop()
            cut.append(float('inf'))
            woe_dif = (d3['woe'][1:d3.shape[0]].reset_index(drop=True) - d3['woe'][:(d3.shape[0]-1)].reset_index(drop=True))
            if (woe_dif>0).sum() == d3.shape[0]-1 or (woe_dif>0).sum() == 0:
                break
            elif allow_u:
                b = 0
                for i in range(1,len(woe_dif)):
                    if woe_dif[i]*woe_dif[i-1] < 0:
                        b += 1
                if b <=1:
                    break
            woe_dif = woe_dif.map(lambda x: np.abs(x))
            d = d3['max'][:-1].reset_index(drop=True)[woe_dif == woe_dif.min()].iloc[0]
            cut.remove(d)
            d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.cut(X, cut,right=True, duplicates='drop')})
            d2 = d1.groupby('Bucket', as_index=False)
            d3 = pd.DataFrame(d2.X.min(), columns=['min'])
            d3['min'] = d2.min().X
            d3['max'] = d2.max().X
            d3['num_bad'] = d2.sum().Y
            d3['total'] = d2.count().Y
            d3['bad_rate'] = d2.mean().Y
            d3['group_rate'] = d3['total'] / (bad + good)
            d3['woe'] = np.log((d3['bad_rate'] / (1 - d3['bad_rate'])) / (bad / float(good)))
            d3['iv'] = (d3['num_bad'] / bad - ((d3['total'] - d3['num_bad']) / good)) * d3['woe']
            iv = d3['iv'].sum()
            d3['iv_sum'] = iv
        d3['range'] = ['('+str(cut[i])+', '+str(cut[i+1])+']' for i in range(len(cut)-1)]
        d3['range'] = d3['range'].astype('str').str.replace('inf]','inf)')
    #将比例低于阈值的分箱进行合箱
    while d3['group_rate'].min() < min_bin:
        mr_index = d3.index[(d3['group_rate'] == d3['group_rate'].min())][0]
        if mr_index == (d3.shape[0]-1):
            cut.remove(cut[-2])
        elif mr_index == 0:
            cut.remove(cut[1])
        else:
            if np.abs(d3['woe'][mr_index-1] - d3['woe'][mr_index]) > np.abs(d3['woe'][mr_index+1] - d3['woe'][mr_index]):
                cut.remove(cut[mr_index])
            else:
                cut.remove(cut[mr_index+1])
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.cut(X, cut,right=True, duplicates='drop')})
        d2 = d1.groupby('Bucket', as_index=False)
        d3 = pd.DataFrame(d2.X.min(), columns=['min'])
        d3['min'] = d2.min().X
        d3['max'] = d2.max().X
        d3['num_bad'] = d2.sum().Y
        d3['total'] = d2.count().Y
        d3['bad_rate'] = d2.mean().Y
        d3['group_rate'] = d3['total'] / (bad + good)
        d3['woe'] = np.log((d3['bad_rate'] / (1 - d3['bad_rate'])) / (bad / float(good)))
        d3['iv'] = (d3['num_bad'] / bad - ((d3['total'] - d3['num_bad']) / good)) * d3['woe']
        iv = d3['iv'].sum()
        d3['iv_sum'] = iv
        d3['range'] = ['('+str(cut[i])+', '+str(cut[i+1])+']' for i in range(len(cut)-1)]
        d3['range'] = d3['range'].astype('str').str.replace('inf]','inf)') 
    d3['ks'] = (d3['total']-d3['num_bad']).cumsum()/float(d3['total'].sum()-d3['num_bad'].sum())-d3['num_bad'].cumsum()/float(d3['num_bad'].sum())       
    d3['label'] = ['bin'+str(i) for i in range(1,len(d3)+1)]
    return d3

#数据分析转换主函数
def WOE_calc(data,
             target = 'target',
             bad = 0,
             min_bin=0.05,
             allow_u = False,
             drop_cols = [],
             iv_thread = 0.02,
             bin_thread = 3,
             add_cols = [],
             out_put = ['IV_table','var_detail','data_woe','data_bin'],
             n_jobs = 4):
    res_df = pd.DataFrame()
    drop_cols += [target]
    for v in data.columns:
        if v not in drop_cols:
            print v
            tmp_df = woe_single(X=data[v],
                                Y=data['target'], 
                                bad = bad,
                                min_bin=min_bin,
                                allow_u = allow_u
                                )
            if tmp_df.shape[0] >= bin_thread:
                tmp_df['variable_name'] = v
                tmp_df = tmp_df[['variable_name','range','label','total','group_rate','num_bad','bad_rate','woe','iv','iv_sum','ks','min','max']]
                res_df = res_df.append(tmp_df)
            else:
                print 'Number of bins Delate: ',v
    res_df = res_df.loc[res_df['iv_sum']>=iv_thread,:]
    res_df['iv_sum_a'] = res_df['iv_sum'] * -1
    res_df['label2'] = res_df['label'].str.replace('bin','').astype(int)
    res_df = res_df.sort_values(['iv_sum_a','label2'],ascending=True)
    del res_df['iv_sum_a']
    del res_df['label2']
    #res_df = res_df[['variable_name','range','label','woe','min','max','total','group_rate','group_rate_t','sum','bad_rate','iv','iv_t','iv_sum','iv_sum_t','ks','ks_t']]    
    IV_table = pd.DataFrame(columns=['variable_name','IV','KS','N_group','mean','mode','min','pct_25','medium','pct_75','max'])
    for v in res_df['variable_name'].drop_duplicates():
        df_tmp = res_df.loc[res_df['variable_name']==v,:]
        IV_tmp = df_tmp['iv'].sum()
        KS_tmp = np.abs(df_tmp['ks']).max()
        N_tmp = df_tmp.shape[0]
        min_tmp = df_tmp['min'].min()
        max_tmp = df_tmp['max'].max()
        mean_tmp = data[v].mean()
        quantile_tmp = data[v].quantile([0.25,0.5,0.75])
        mode_tmp = data[v].value_counts().sort_index(ascending=False).index[0]
        IV_table.loc[IV_table.shape[0],:] = [v,IV_tmp,KS_tmp,N_tmp,mean_tmp,mode_tmp,min_tmp,quantile_tmp.iloc[0],quantile_tmp.iloc[1],quantile_tmp.iloc[2],max_tmp]     
    Data_outupt = pd.DataFrame()
    #print train_X.shape
    for c in add_cols+[target]:
        Data_outupt[c] = data[c]
    if 'data_woe' in out_put:
        print 'Now start to get WOE data.'
        for v in IV_table.variable_name:
            Data_outupt['WOE_'+v] = bin_seq2_n(data[v],res_df.loc[res_df['variable_name']==v,:],seq_dir=seq_dir,label='woe',other_fill=-999)
    if 'data_bin' in out_put:
        print 'Now start to get range data.'
        for v in IV_table.variable_name:
            Data_outupt['bin_'+v] = bin_seq2_n(data[v],res_df.loc[res_df['variable_name']==v,:],seq_dir=seq_dir,label='range',other_fill='no range for the value !')
    if 'data_woe' in out_put or 'data_bin' in out_put:
        return IV_table,res_df,Data_outupt
    else:
        return IV_table,res_df

#相关性过滤函数
def corr_filter(IV_table, data, cor_thr = 0.7):
    var_rm = []
    var_sel= []
    var_list = IV_table['variable_name'][1:].tolist()
    for v in IV_table['variable_name']:
        if v not in var_rm:
            for v2 in var_list:
                if np.abs(data[['WOE_'+v,'WOE_'+v2]].corr().iloc[0,1]) > cor_thr:
                    var_list.remove(v2)
                    var_rm.append(v2)
            var_sel.append(v)
    return var_sel

#前向逐步回归
def stepwise_logistic_forward(y, data, var_list,alpha=0.05):
    varlist_step = []
    in_var = []
    out_var =  [v for v in var_list]
    while True:
        if len(out_var) == 0:
            break
        min_p = 10
        min_var = ''
        for v in out_var:
            model_tmp = sm.GLM(data[y],sm.add_constant(data[in_var + [v]]),family=sm.families.Binomial()).fit()
            p_tmp = model_tmp.pvalues[v]
            if min_p > p_tmp:
                min_p = p_tmp
                min_var = v
        in_var.append(min_var)
        print 'add var: ' + min_var
        out_var.remove(min_var)
        model_tmp = sm.GLM(data[y],sm.add_constant(data[in_var]),family=sm.families.Binomial()).fit()
        p_list = model_tmp.pvalues
        rm_var = [a for a in p_list.index[p_list>alpha]]
        if 'const' in rm_var:
            rm_var.remove('const')
        rm_var = [a for a in p_list.index[p_list>alpha]]
        for v in rm_var:
            out_var.append(v)
            in_var.remove(v)
            print 'remove var: ' + v
        in_var.sort()
        if in_var in varlist_step:
            break
        varlist_step.append([v for v in in_var])
    return in_var

#辅助函数：将Series转为dataframe
def df_1d_create(x,name,id_name):
    return pd.DataFrame({id_name:x.index,name:x})

#ks计算函数
def evalKS(preds, Y):       # written by myself
    fpr, tpr, thresholds = roc_curve(Y, preds)
    #print 'KS_value', (tpr-fpr).max()
    return 'KS_value', np.abs(tpr-fpr).max()

#woe预测分数函数
def woe_score_predict(x,card):
    x['id_tmp'] = range(x.shape[0])
    res = pd.DataFrame({'id_tmp':x['id_tmp']})
    for v in card['variable_name'].unique():
        card_tmp = card.loc[card['variable_name']==v,['woe','score']]
        card_tmp['woe'] = card_tmp['woe'].round(6)
        tmp = x[['id_tmp','WOE_'+v]].rename(columns={'WOE_'+v:'woe'})
        tmp['woe'] = tmp['woe'].round(6)
        tmp = pd.merge(tmp,card_tmp,on='woe').drop_duplicates()
        tmp = tmp.rename(columns={'score':'s_'+v})[['id_tmp','s_'+v]]
        res = pd.merge(res,tmp,on='id_tmp',how='left')
    res = res.sort_values('id_tmp')
    del res['id_tmp']
    del x['id_tmp']
    #res['score'] = res.apply(lambda x: sum(x)-x['id_tmp'],axis=1)
    res['score'] = res.apply(lambda x: sum(x),axis=1)
    return res

def Credit_model_build(IV_table,
                       var_detail,
                       data,
                       target = 'target',
                       var_limit = [],
                       var_drop = [],
                       Info_add = [],
                       cor_thr = 0.7,
                       alpha = 0.05,
                       method = 'forward',
                       simple = True,
                       basic_score = 600,
                       basic_rate = 4,
                       per_score = 50,
                       score_cut = 'cut'):
    IV_table_model = IV_table.loc[IV_table.variable_name.map(lambda x: x not in var_drop)]
    if len(var_limit) > 0:
        IV_table_model = IV_table_model.loc[IV_table_model.variable_name.map(lambda x: x in var_limit)]
    IV_table_model = IV_table_model.reset_index(drop=True)
    print 'Now start to correlation filter: '
    var_corr = corr_filter(IV_table_model, data, cor_thr = cor_thr)
    var_corr = ['WOE_'+v for v in var_corr]
    print 'Now start Stepwise Logistic Model procedure: '
    var_for_model = stepwise_logistic_forward(target, data, var_corr,alpha=0.05)
    logistic_model = sm.GLM(data[target],sm.add_constant(data[var_for_model]),family=sm.families.Binomial()).fit()
    
    print 'Now collecting the results of Model: '
    model_res = {}
    #系数结果
    model_res['conf_table'] = logistic_model.summary2().tables[1]
    #入模变量汇总
    model_res['var_summary'] = IV_table_model.loc[IV_table_model['variable_name'].map(lambda x:'WOE_'+x in var_for_model),:].reset_index(drop=True)
    #入模变量细节
    model_res['var_detail'] = var_detail.loc[var_detail['variable_name'].map(lambda x: 'WOE_'+x in var_for_model)]
    #评分卡结果
    b, o, p = basic_score, basic_rate, per_score
    conf_list = model_res['conf_table']['Coef.']
    detail_card = model_res['var_detail'].loc[model_res['var_detail']['variable_name'].map(lambda x:'WOE_'+x in conf_list.index.tolist())]
    detail_card = detail_card[['variable_name','group_rate','bad_rate','woe','range','label']]
    num_var = conf_list.shape[0]-1
    detail_card['score'] = detail_card.apply(lambda x: round(p/np.log(2) * (-conf_list.loc['const']/num_var - conf_list.loc['WOE_'+x['variable_name']]*x['woe']) + (b-p*np.log(o)/np.log(2))/num_var) ,axis=1)
    model_res['score_card'] = detail_card
    model_res['model_corr_mat'] = data[model_res['conf_table'].index[1:].tolist()].corr()
    model_res['var_corr_filter'] = pd.DataFrame({'variable_name': var_corr})
    
    #训练集分数
    score_train = woe_score_predict(data, detail_card)
    for c in Info_add:
        score_train[c] = data[c]
    model_res['train_score'] = score_train
    #训练集分数cut
    if score_cut == 'cut':
        score_seq = [score_train['score'].min() + i * (score_train['score'].max()-score_train['score'].min())/10. for i in range(11)]
    else:
        score_seq = score_train['score'].quantile([i/10. for i in range(11)])
    score_bin = pd.cut(score_train['score'],score_seq,duplicates='drop',include_lowest=True).astype(str)
    score_bin_res = score_bin.value_counts().sort_index()
    score_bin_res = df_1d_create(score_bin_res,'num','score_bin')
    tmp = data[target].groupby(score_bin).sum()
    tmp = df_1d_create(tmp,'bad_num','score_bin')
    score_bin_res = pd.merge(score_bin_res,tmp,on='score_bin',how='left')
    score_bin_res['good_num'] = score_bin_res.apply(lambda x:x['num']-x['bad_num'], axis=1)
    score_bin_res['total_rate'] = score_bin_res['num'].map(lambda x: x/float(data.shape[0]))
    score_bin_res['bad_rate'] = score_bin_res.apply(lambda x: x['bad_num']*1.0/x['num'],axis=1) 
    score_bin_res['cumsum_bad_rate'] = score_bin_res['bad_num'].cumsum().map(lambda x: x/float(data[target].sum()))
    score_bin_res['cumsum_good_rate'] = score_bin_res['good_num'].cumsum().map(lambda x: x/float(data.shape[0]-data[target].sum()))
    score_bin_res['KS'] = score_bin_res['cumsum_bad_rate'] - score_bin_res['cumsum_good_rate']
    model_res['score_density_train_cut'] = score_bin_res[['score_bin','num','bad_num','good_num','total_rate','bad_rate','cumsum_bad_rate','cumsum_good_rate','KS']]
    
    train_pre = logistic_model.predict(sm.add_constant(data[var_for_model]))
    accuracy = accuracy_score(data[target],train_pre.map(lambda x: round(x)))
    precision = precision_score(data[target],train_pre.map(lambda x: round(x)))
    recall = recall_score(data[target],train_pre.map(lambda x: round(x)))
    auc = roc_auc_score(data[target],train_pre)
    ks = evalKS(train_pre, data[target])[1]
    cut_ks = model_res['score_density_train_cut']['KS'].max()
    model_res['model_summary'] = pd.DataFrame({'train':[accuracy,precision,recall,auc,ks,cut_ks],
                                               'var':['accuracy','precision','recall','auc','ks','cut_ks']})[['var','train']]
    return model_res

#分数分箱函数
def bin_seq2_score(X,range_df,seq_dir=seq_dir,other_fill='unknown_bins'):  
    def range_in(x,r_tmp):
        if type(r_tmp) == list:
            return (seq_dir[r_tmp[-1]](x,r_tmp[-2]))
        else:
             return (x == x.dtype.type(r_tmp)) | (x.astype(type(r_tmp)) == r_tmp)       
    X_np = X.values
    X_res = np.array([other_fill]*X.shape[0])
    for r in range_df['score_bin']:
        try:
            r_tmp = str(r).replace(' ','').split(',')
            r_tmp = [r_tmp[0][0]] + [float(r_tmp[0].replace(r_tmp[0][0],''))] + [float(r_tmp[1].replace(r_tmp[1][-1],''))] + [r_tmp[1][-1]]
        except:
            r_tmp = r
        X_res[range_in(X_np,r_tmp) & (X_res == other_fill)] = range_df['score_bin'][range_df['score_bin']==r].iloc[0] 
    return pd.Series(X_res)

def Credit_model_test(model_res,
                      test_data,
                      Info_add = [],
                      target = 'target',
                      use_target = True):
    test_score = pd.DataFrame(index = range(test_data.shape[0]))
    for c in Info_add:
        test_score[c] = test_data[c]
    test_score['score'] = 0
    for v in model_res['var_summary']['variable_name']:
        range_df = model_res['score_card'].loc[model_res['score_card']['variable_name'] == v,:].reset_index(drop=True)
        test_score['s_'+v] = bin_seq2_n(test_data[v],
                                      range_df,
                                      seq_dir=seq_dir,
                                      label='score',
                                      other_fill=0)
        test_score['score'] = test_score.apply(lambda x: x['score'] + x['s_'+v], axis=1)
    score_bin = bin_seq2_score(test_score['score'],
                                   model_res['score_density_train_cut'],
                                   seq_dir=seq_dir,
                                   other_fill='still_unknown_bins')
    score_bin_res = df_1d_create(score_bin.value_counts().sort_index(),'num','score_bin')
    if use_target:
        tmp = test_data[target].groupby(score_bin).sum()
        tmp = df_1d_create(tmp,'bad_num','score_bin')
        score_bin_res = pd.merge(score_bin_res,tmp,on='score_bin',how='left')
        score_bin_res['good_num'] = score_bin_res.apply(lambda x:x['num']-x['bad_num'], axis=1)
        score_bin_res['total_rate'] = score_bin_res['num'].map(lambda x: x/float(test_data.shape[0]))
        score_bin_res['bad_rate'] = score_bin_res.apply(lambda x: x['bad_num']*1.0/x['num'],axis=1) 
        score_bin_res['cumsum_bad_rate'] = score_bin_res['bad_num'].cumsum().map(lambda x: x/float(test_data[target].sum()))
        score_bin_res['cumsum_good_rate'] = score_bin_res['good_num'].cumsum().map(lambda x: x/float(test_data.shape[0]-test_data[target].sum()))
        score_bin_res['KS'] = score_bin_res['cumsum_bad_rate'] - score_bin_res['cumsum_good_rate']
    else:
        score_bin_res['total_rate'] = score_bin_res['num'].map(lambda x: x/float(test_data.shape[0]))
    score_bin_res = pd.merge(score_bin_res,model_res['score_density_train_cut'][['score_bin','total_rate']].rename(columns = {'total_rate':'total_rate_train'}),
                             on='score_bin',how='right')
    score_bin_res['PSI'] = score_bin_res.apply(lambda x: np.log(x['total_rate']/x['total_rate_train'])*(x['total_rate'] - x['total_rate_train']), axis=1)
    
    test_res = {'test_score':test_score,'score_density_test_cut':score_bin_res}
    return test_res    

def Credit_res_save(model_res, file_name):
    writer = pd.ExcelWriter(file_name + '.xlsx')
    for k in model_res.keys():
        model_res[k].to_excel(writer,sheet_name = k)
    writer.save()

def Credit_res_load(file_name):
    model_ex = pd.ExcelFile(file_name+'.xlsx')
    model_res = {}
    for k in model_ex.sheet_names:
        model_res[k] = pd.read_excel(file_name+'.xlsx',sheet_name = k)
    return model_res

####################################test###################################

import os
os.chdir(os.getcwd()+'/test')
#读取credit比赛数据
credit_data = pd.read_csv('cs-training.csv').iloc[:,1:]

#数据概况，缺失值默认填充为-1
credit_data.describe()
credit_data = credit_data.fillna(-1.0)
credit_data = credit_data.rename(columns = {'SeriousDlqin2yrs':'target'})

#划分训练和数据集（需要重置index）
credit_data_train,credit_data_test = train_test_split(credit_data,test_size = 0.3,random_state=66)
credit_data_train,credit_data_test = credit_data_train.reset_index(drop=True), credit_data_test.reset_index(drop=True)

#数据清洗流程，得到IV分析结果、字段分箱细节和转化成woe以及range的数据
#设定坏人标签为1，标签列为‘target’，占比最小的分箱是0.05（否则会被合箱），变量最低的分箱数为3，允许非严格单调（U型）
credit_IV, credit_detail, credit_woe = WOE_calc(credit_data_train,
                                                target = 'target',
                                                bad = 1,
                                                min_bin=0.05,
                                                allow_u = True,
                                                iv_thread = 0.02,
                                                bin_thread = 3)

#评分卡建模流程
#输入数据清洗流程的3个结果，相关性阈值为0.7，显著性阈值P为0.1，分箱方式为等量10分箱（qcut）
credit_model = Credit_model_build(credit_IV,
                                  credit_detail,
                                  credit_woe,
                                  target = 'target',
                                  cor_thr = 0.7,
                                  alpha = 0.1,
                                  score_cut = 'qcut')

print credit_model['conf_table'] #查看模型系数信息
print credit_model['score_card'] #查看评分卡表
print credit_model['var_summary'] #查看入模字段分析结果
print credit_model['score_density_train_cut'] #查看分数分箱汇总结果

#评分卡对测试数据的测试
credit_test = Credit_model_test(credit_model,
                                credit_data_test)

print credit_test['score_density_train_cut']#查看测试集分数分箱汇总结果

#评分卡模型存储和加载
Credit_res_save(credit_model,'credit_model_card')
credit_model_2 = Credit_res_load('credit_model_card')














