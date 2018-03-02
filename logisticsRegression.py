# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 09:24:15 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:28:25 2017
 
@author: toby
 
CSV数据结构，第一列为数值，第二列为二分类型
"""
import csv
import numpy as np
import pandas as pd
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial
import matplotlib.pyplot as plt
import seaborn as sns
#中文字体设置
from matplotlib.font_manager import FontProperties
font=FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=14)

#该函数的其他的两个属性"notebook"和"paper"却不能正常显示中文
sns.set_context('poster')
 
fileName="同盾多头借贷与同盾分数回归分析.csv"
reader = csv.reader(open(fileName))
 
 
#获取数据，类型：阵列
def getData():
    '''Get the data '''
     
    inFile = '同盾多头借贷与同盾分数回归分析.csv'
    data = np.genfromtxt(inFile, skip_header=1, usecols=[0, 1],
                                    missing_values='NA', delimiter=',')
    # Eliminate NaNs 消除NaN数据
    data1 = data[~np.isnan(data[:, 1])]
    return data1
     
     
def prepareForFit(inData):
    ''' Make the temperature-values unique, and count the number of failures and successes.
    Returns a DataFrame'''
     
    # Create a dataframe, with suitable columns for the fit
    df = pd.DataFrame()
    #np.unique返回去重的值
    df['同盾分数'] = np.unique(inData[:,0])
    df['同盾多头借贷命中'] = 0
    df['同盾多头借贷未命中'] = 0
    df['total'] = 0
    df.index = df.同盾分数.values
     
    # Count the number of starts and failures
    #inData.shape[0] 表示数据多少
    for ii in range(inData.shape[0]):
        #获取第一个值的温度
        curTemp = inData[ii,0]
        #获取第一个值的值，是否发生故障
        curVal  = inData[ii,1]
        df.loc[curTemp,'total'] += 1
        if curVal == 1:
            df.loc[curTemp, '同盾多头借贷命中'] += 1
        else:
            df.loc[curTemp, '同盾多头借贷未命中'] += 1
    return df
 
     
#逻辑回归公式
def logistic(x, beta, alpha=0):
    ''' Logistic Function '''
    #点积，比如np.dot([1,2,3],[4,5,6]) = 1*4 + 2*5 + 3*6 = 32
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))   
 
     
#不太懂   
def setFonts(*options):
        return   
#绘图   
def Plot(data,alpha,beta,picName):
    #阵列，数值
    array_values = data[:,0]
    #阵列，二分类型
    array_type = data[:,1]
 
    plt.figure(figsize=(15,15))
    setFonts()
    #改变指定主题的风格参数
    sns.set_style('darkgrid')
    #numpy输出精度局部控制
    np.set_printoptions(precision=3, suppress=True)
    plt.scatter(array_values, array_type, s=200, color="k", alpha=0.5)
    #获x轴列表值，同盾分数
    list_values = [row[0] for row in reader][1:]
    list_values = [int(i) for i in list_values]
    #获取列表最大值和最小值
    max_value=max(list_values)
    print("max_value:",max_value)
    min_value=min(list_values)
    print("min_value:",min_value)
    #最大值和最小值留有多余空间
    x = np.arange(min_value, max_value)
    y = logistic(x, beta, alpha) 
    plt.hold(True)
    plt.plot(x,y,'r')
    #设置y轴坐标刻度
    plt.yticks([0, 1])
    #plt.xlim()返回当前的X轴绘图范围
    plt.xlim([min_value,max_value])
    outFile = picName
    plt.ylabel("同盾多头借贷命中概率",fontproperties=font)
    plt.xlabel("同盾分数",fontproperties=font)
    plt.title("逻辑回归-同盾分数VS同盾多头借贷命中概率",fontproperties=font)
    #产生方格
    plt.hold(True)
    #图像外部边缘的调整
    plt.tight_layout
    plt.show(outFile)
     
     
#用于预测逻辑回归概率
def Prediction(x):
    y = logistic(x, beta, alpha)   
    print("probability prediction:",y)
'''
Prediction(80)
probability prediction: 0.872046286637

Prediction(100)
probability prediction: 0.970179520648

'''
     
#获取数据   
inData = getData()
#得到频率计算后的数据
dfFit = prepareForFit(inData)   
#Generalized Linear Model 建立二项式模型
model = glm('同盾多头借贷未命中 +同盾多头借贷命中 ~ 同盾分数', data=dfFit, family=Binomial()).fit()   
print(model.summary())
chi2=model.pearson_chi2
'''Out[37]: 46.893438309853522  分数越小，p值越大，H0成立，模型越好'''
print("the chi2 is smaller,the model is better") 

alpha = model.params[0]
beta = model.params[1]
 
Plot(inData,alpha,beta,"logiscti regression")

#测试
Prediction(20)
Prediction(60)
Prediction(80)










