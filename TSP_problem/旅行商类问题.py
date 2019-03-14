#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:57:13 2018

@author: jq_tongdun
"""

import numpy as np
import random
import copy
import matplotlib.pyplot as plt

np.random.seed(20)
all_points = np.random.randint(0,100,(30,2))
#all_points_num = np.array(range(30))

#距离函数
def dist_calc(points):
    dist_mat = np.zeros((points.shape[0],points.shape[0]))
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
           dist_mat[i,j] = np.sqrt(np.sum((points[i,:]-points[j,:])**2)) 
    return dist_mat

#示例距离矩阵
exam_mat = dist_calc(all_points)

#################遗传算法尝试####################

#定义遗传算法类
class gen_alg(object):
    #传入参数
    def __init__(self,dist_mat,start_point,end_point,points=None,init_time=100,
                 pop_num=100,p_cros=0.5,p_vari=0.1,p_best=0.1):
        self.dist_mat = dist_mat
        self.start_point = start_point
        self.end_point = end_point
        self.init_time = init_time
        self.pop_num = pop_num
        self.p_cros = p_cros
        self.p_vari = p_vari
        self.p_best = p_best
        self.points = points
    #目标函数定义（以欧氏距离和为准）
    def aim_func(self,order):
        aim = 0
        for i in range(len(order)-1):
            aim += self.dist_mat[order[i],order[i+1]]
            #print aim
        return aim
    #随机生成单个顺序
    def rand_order(self):
        start = self.start_point
        end = self.end_point
        org_order = range(self.dist_mat.shape[0])
        org_order.remove(start)
        org_order.remove(end)
        random.shuffle(org_order)
        new_order = [start] + org_order + [end]
        return new_order
    #随机生成的种群样本
    def get_org_population(self,pop_num):
        pop_init = []
        for i in range(pop_num):
            pop_init.append(self.rand_order())
        return pop_init
    #两个个体交叉函数
    def cross_func(self,order1,order2):
        order1 = copy.copy(order1)
        order2 = copy.copy(order2)
        order1.remove(self.start_point)
        order1.remove(self.end_point)
        order2.remove(self.start_point)
        order2.remove(self.end_point)
        num_cros = random.randint(1,(len(order1)-1))
        sel_cros = random.sample(range(len(order1)),num_cros)
        order1_cp,order2_cp = copy.copy(order1),copy.copy(order2)
        for i in sel_cros:
            order1_cp[i] = order2[i]
            order1_cp[order1.index(order2[i])] = order1[i]
            order2_cp[i] = order1[i]
            order2_cp[order2.index(order1[i])] = order2[i]   
            order1 = copy.copy(order1_cp)
            order2 = copy.copy(order2_cp)
        return [self.start_point]+order1_cp+[self.end_point],[self.start_point]+order2_cp+[self.end_point]
    #变异函数   
    def vari_func(self,order):
        num_vari = random.randint(1,(len(order)/2))
        sel_vari = random.sample(order[1:(len(order)-1)],num_vari)
        order_cp = copy.copy(order)
        for i in sel_vari:
            value_tmp = random.sample(order[1:(len(order)-1)],1)[0]
            order_cp[i] = value_tmp
            order_cp[order.index(value_tmp)] = order[i]
            order = copy.copy(order_cp)
        return order_cp
    #训练主函数
    def gen_alg(self):
        #初始化：产生初代群落样本
        init_pop = self.get_org_population(self.pop_num)
        for i in range(self.init_time):
            #按目标值排序
            aims_tmp = [self.aim_func(o) for o in init_pop]
            index_aims = range(len(aims_tmp))
            index_aims.sort(key=lambda i: aims_tmp[i])
            print "for the init %s, the best aim value is %s" % (i,aims_tmp[index_aims[0]])
            #分别抽取用于交叉的优秀样本和可以直接进入下一代的最有样本群落
            good_pop = [init_pop[j] for j in index_aims[:int(self.pop_num*self.p_cros)]]
            best_pop = good_pop[:int(self.pop_num*self.p_best)]
            #绘图
            if (i+1)%10 == 0:
                if self.points is not None:
                    plt.plot(self.points[best_pop[0],0], self.points[best_pop[0],1],
                             marker='o', mec='r', mfc='w')
                    plt.show()  
                #判断迭代次数是否结束
                if i == self.init_time-1:
                    print "Finally the best aim value is %s" % aims_tmp[index_aims[0]]
                    print "And the best order is: "
                    print best_pop[0]
                    self.best_pop = best_pop[0]
                    return best_pop[0]
            #交叉过程
            random.shuffle(good_pop)
            cros_pop = [0]*len(good_pop)
            #print len(cros_pop)
            for l in range(0,len(good_pop),2):
                cros_tmp = self.cross_func(good_pop[l],good_pop[l+1])
                cros_pop[l] = cros_tmp[0]
                cros_pop[l+1] = cros_tmp[1]
            #交叉和最优继承的样本以外再增加一些随机样本，作为下一代群落
            init_pop = init_pop+cros_pop+best_pop
            vari_idx = random.sample(range(len(init_pop)),int(self.pop_num*self.p_vari))
            for m in vari_idx:
                 init_pop[m] = self.vari_func(init_pop[m])
            #变异过程     
            init_pop = init_pop + self.get_org_population(self.pop_num-int(self.pop_num*self.p_cros)-int(self.pop_num*self.p_best))
        
#定义类的时候需要传入一个距离矩阵，起始点和终点，以及训练参数：
'''
init_time:迭代次数
pop_num:每次迭代的种群数量
p_cros:选出用于交配样本的比例
p_vari:变异样本的比例
p_best:直接进入下次迭代最优样本的比例

'''
a = gen_alg(exam_mat,0,29,points=all_points,pop_num=3000,init_time=200,p_cros=0.6,
            p_vari=0.1,p_best=0.1)   
#a.aim_func(exam_order)    
#a.rand_order()
#b = a.get_org_population(10)
#a.cross_func(b[0],b[1])
#a.vari_func(b[0])
a.gen_alg()


######################模拟退火算法尝试#####################

class Sim_Ann(object):
    #传入参数
    def __init__(self,dist_mat,start_point,end_point,points=None,t0=100,t1=1,
                 alpha=0.99,p_two_exchange=0.5):
        self.dist_mat = dist_mat
        self.start_point = start_point
        self.end_point = end_point
        self.points = points
        self.t0 = t0
        self.t1 = t1
        self.alpha = alpha
        self.P_two_exchange = p_two_exchange
    #目标函数定义（以欧氏距离和为准）
    def aim_func(self,order):
        aim = 0
        for i in range(len(order)-1):
            aim += self.dist_mat[order[i],order[i+1]]
            #print aim
        return aim
    #随机生成单个顺序
    def rand_order(self):
        start = self.start_point
        end = self.end_point
        org_order = range(self.dist_mat.shape[0])
        org_order.remove(start)
        org_order.remove(end)
        random.shuffle(org_order)
        new_order = [start] + org_order + [end]
        return new_order
    #双交换函数
    def two_exchange(self,order):
        while True:
            loc1 = np.random.randint(1,len(order)-1)
            loc2 = np.random.randint(1,len(order)-1)
            if loc1 != loc2:
                break
        order_new = copy.copy(order)
        order_new[loc1],order_new[loc2] = order_new[loc2],order_new[loc1]
        return order_new
    #三交换函数
    def three_exchange(self,order):
        while True:
            loc1 = np.random.randint(1,len(order)-1)
            loc2 = np.random.randint(1,len(order)-1)
            loc3 = np.random.randint(1,len(order)-1)
            if loc1 != loc2 and loc1 != loc3 and loc2 != loc3:
                break
        order_new = copy.copy(order)
        order_new[loc1],order_new[loc2],order_new[loc3] = order_new[loc2],order_new[loc3],order_new[loc1]
        return order_new
    #训练主函数
    def Sim_Ann_fit(self):
        t = self.t0
        order_init = self.rand_order()
        l=0
        while t >= self.t1:
            if random.random() <= self.P_two_exchange:
                order_new = self.two_exchange(order_init)
            else:
                order_new = self.three_exchange(order_init)
            aim_new = self.aim_func(order_new)
            aim_old = self.aim_func(order_init)
            if aim_new < aim_old:
                order_init = order_new
            else:
                if np.exp(-(aim_new-aim_old)/float(t)) > random.random():
                    order_init = order_new
            if l%1000 == 0:
                print 'Now the temperature is ' + str(t)
                print 'And the aim value is: ' + str(self.aim_func(order_init))   
                if self.points is not None:
                    plt.plot(self.points[order_init,0], self.points[order_init,1],
                             marker='o', mec='r', mfc='w')
                    plt.show() 
            t = t*self.alpha
            l += 1
        self.final_order = order_init
        print 'Finally the Order is: '
        print self.final_order
        print 'And the aim value is: ' + str(self.aim_func(order_init))
        plt.plot(self.points[order_init,0], self.points[order_init,1],
                             marker='o', mec='r', mfc='w')
        plt.show() 
        return order_init
            
b = Sim_Ann(exam_mat,0,29,points=all_points,t0=10000,alpha=0.9999)
#b.two_exchange(a)
#b.three_exchange(a)
b.Sim_Ann_fit()


