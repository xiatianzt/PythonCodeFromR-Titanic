
# -*- coding: utf-8 -*-
'''
Created on 2016��5��3��

@author: zt
'''

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.ensemble import RandomForestRegressor

def fsizesur_bar_chart(): 
    global full
    ###用groupby的方式构建透视表
#     fsizeandsur = full.groupby(['Fsize','Survived']).size()
#     survived = fsizeandsur.loc[:,1].copy()
#     unsurvived = fsizeandsur.loc[:,0].copy()
#     survived[8] = 0;survived[11]=0
    
    ###用corsstab或者pivot_table构建透视表，相当于r语言中的table()函数
    fsizeandsur = pd.crosstab(full['Fsize'], full['Survived'])
    survived = fsizeandsur.loc[:,1]
    unsurvived = fsizeandsur.loc[:,0]
    
    plt.subplot(1,1,1)
    ind = np.arange(9)
    width = 0.35
    #画出两个系列的柱  
    plt.bar(ind,survived,width,color='r',label='Survived')
    plt.bar(ind+width,unsurvived,width,color='y',label='unSurvived')
    
    plt.xlabel('Family Size')
    plt.ylabel('count')
    plt.legend()
    plt.xticks(ind+width,(1,2,3,4,5,6,7,8,11))
    plt.show()

def fsizedsur_mosaic_plot():
    global full
    
#     fsizedandsur = pd.crosstab(full['FsizeD'], full['Survived'])

    mosaic(full,['FsizeD','Survived'])
    plt.show()

def empcfa_box_plot():
    global full
    
    fullcopy = full[['Embarked','Pclass','Fare']].drop(full.index[[61,829,1043]]).copy()
    
    #####使用系统提供的功能画出boxplot
#     empcfa =  fullcopy.groupby(['Embarked','Pclass'])
#     empcfa.boxplot( column='Fare',subplots=True,return_type='dict')
#     plt.show()
    
    #####
    empcfa =  fullcopy.groupby(['Embarked','Pclass']).groups
    plt.subplot(1,3,1)
    plt.boxplot([fullcopy.loc[empcfa[('C',1)],'Fare'],fullcopy.loc[empcfa[('Q',1)],'Fare'],
                 fullcopy.loc[empcfa[('S',1)],'Fare']],vert=True,patch_artist=True)
    plt.subplot(1,3,2)
    plt.boxplot([fullcopy.loc[empcfa[('C',2)],'Fare'],fullcopy.loc[empcfa[('Q',2)],'Fare'],
                 fullcopy.loc[empcfa[('S',2)],'Fare']],vert=True,patch_artist=True)
    plt.subplot(1,3,3)
    plt.boxplot([fullcopy.loc[empcfa[('C',3)],'Fare'],fullcopy.loc[empcfa[('Q',3)],'Fare'],
                 fullcopy.loc[empcfa[('S',3)],'Fare']],vert=True,patch_artist=True)
    plt.show()
    
def fare3s_density():
    global full
    
    fare3s = full.loc[full['Pclass']==3,].loc[full['Embarked']=='S','Fare']
    fare3s.plot(kind='density')
    plt.show()

def setMissingAges():
    global full
    
    age_full = full[['Age','Embarked','Fare', 'Parch', 'SibSp', 'Title','Pclass','Fsize']].copy()
    
#     age_full.loc[age_full['Embarked'] == 'S','Embarked']=1
#     age_full.loc[age_full['Embarked'] == 'C','Embarked']=2
#     age_full.loc[age_full['Embarked'] == 'Q','Embarked']=3
#     age_full.loc[age_full['Title']=="Miss","Title"] = 1
#     age_full.loc[age_full['Title']=="Mrs","Title"] = 2
#     age_full.loc[age_full['Title']=="Mr","Title"] = 3
#     age_full.loc[age_full['Title']=="Master","Title"] = 4
#     age_full.loc[age_full['Title']=="RareTitle","Title"] = 5

    age_full['Embarked'] = pd.factorize(age_full['Embarked'])[0]
    age_full['Title'] = pd.factorize(age_full['Title'])[0]
    ####可以用下面方法来获取哪些行是null
    known = age_full[(age_full['Age'].notnull())]
    unknown = age_full.loc[(age_full['Age'].isnull())]
    
    yknown = known['Age']
    Xknown = known[['Embarked','Fare', 'Parch', 'SibSp', 'Title','Pclass','Fsize']]
    ####使用随机森林训练数据
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(Xknown, yknown)
    #####预测并填充
    predicted = rtr.predict(unknown[['Embarked','Fare', 'Parch', 'SibSp', 'Title','Pclass','Fsize']])    
    full.loc[(age_full['Age'].isnull()),'Age'] = predicted
    #####画出年龄分布直方图
#     plt.subplot(1,1,1)
#     plt.hist(full['Age'])
#     plt.show()
    
def agesur_hist_plot():
    global full
    
#     agesur = pd.crosstab(full['Age'], full['Survived'])
    
    ages0 = full.loc[full['Survived']==0,]
    ages1 = full.loc[full['Survived']==1,]
    
    ages0male = ages0.loc[full['Sex']=='male','Age']
    ages0female = ages0.loc[full['Sex']=='female','Age']
    ages1male = ages1.loc[full['Sex']=='male','Age']
    ages1female = ages1.loc[full['Sex']=='female','Age']
    
    plt.subplot(1,2,1)
    plt.hist([ages0male,ages1male],histtype='stepfilled',stacked=True)
    plt.subplot(1,2,2)
    plt.hist([ages0female,ages1female],histtype='stepfilled',stacked=True)
    plt.show()
    
def getDataSet():
    
    global full
    
    train = pd.read_csv("data\\train.csv")
    test = pd.read_csv("data\\test.csv")
    #合并两个表
    full = train.append(test, ignore_index=True)
    full["Title"] = full["Name"].map(lambda x: re.sub('(.*,.)|(\..*)','', x))
    #下面这个命令可以得到每个Title的总大小
    ######
#     print full[['Sex',"Title"]].groupby("Title").count()
    #####将Title中的一些稀有名称用统一符号代替
    rare_title = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
    full.loc[full['Title']=="Mlle","Title"] = 'Miss'
    full.loc[full['Title']=="Ms","Title"] = 'Miss'
    full.loc[full['Title']=="Mme","Title"] = 'Mrs'
    #注意，lambda匿名函数也接受if else语句，但是if else必须连在一起，x1 if condition else x2，x1是满足if的值，x2是不满足if的值
    full['Title'] = full['Title'].map(lambda x : 'RareTitle' if x in rare_title else x)
    ######
#     print full[['Sex',"Title"]].groupby("Title").count()
    
    #可以通过下面这个句子得到每个性别下每个title的大小，类型是Series，例如用x['female']['Dr']即可访问到每个的个数
#     print full.groupby(["Sex","Title"]).size()
    #提取surname，提取，号前面的为suranme
    full['Surname'] = full['Name'].map(lambda x: re.split('[,.]',x)[0])
    #######打印surname中不同surname的个数
#     print full['Surname'].unique().size
    
    full['Fsize'] = full['SibSp'] + full['Parch'] + 1
    full['Family'] = full['Surname'] + '_' + str(full['Fsize'])
    
    #绘制family size和survived变量的柱状图
#     fsizesur_bar_chart()
    #经过分析发现fsize == 1，未存活比存活多，2~4存活比未存活多，5~未存活比存活多，所以分成三部分
    full.loc[full['Fsize']==1,'FsizeD'] = 'singleton'
    full.loc[full['Fsize']>1,'FsizeD'] = 'small'
    full.loc[full['Fsize']>=5,'FsizeD'] = 'large'
    
    #绘制mosaic plot
#     fsizedsur_mosaic_plot()
    
    #填充Cabin的空值,这一部在原文中没有，提取舱面，Cabin的第一个字母，
    full['Cabin'] = full['Cabin'].fillna('U0')
    full['Deck'] = full['Cabin'].map(lambda x: list(str(x))[0])
    
    #有两个值空缺，embarked
#     print full['Embarked'].isnull().value_counts()

    #####分析embarked，pclass，fare的关系
#     empcfa_box_plot()
    
    #将两个缺失的embarked填充为C
    full['Embarked']  = full['Embarked'].fillna('C')
    
    ####Fare缺失一个1043，fare和pclass和embarked相关
#     fare3s_density()
    
    ######取class为3，embarked为s的中位数来填充 空值
    full.loc[1043,'Fare'] = full.loc[full['Pclass']==3,].loc[full['Embarked']=='S','Fare']\
                                .median()
    
    #####计算age中缺失的数量
#     print full['Age'].isnull().value_counts()

    ####使用mice来填充缺失数据在这个版本里的stats里似乎没有，http://gsocfrankcheng.blogspot.jp/
    ####可以推荐一个包，https://github.com/hammerlab/fancyimpute，但我懒得安装了，安装了下，各种依赖……
    ####用决策树来决定如何填充Age的值
    setMissingAges()
    ####画出存活和未存活的人在不同年龄段下的分布
#     agesur_hist_plot()
    
    
    #####添加新特征，标记这个人是否是孩子
    full.loc[full['Age'] <18,'Child'] = 1
    full.loc[full['Age'] >=18,'Child'] = 0
    #打印儿童的存活个数
#     print full.groupby(["Survived","Child"]).size()

    #####添加新特征，标记这个人是否是母亲
    full['Mother'] = 0
    full.loc[(full.Sex == 'female') & (full.Parch > 0) & (full.Age>18)
             & (full.Title != 'Miss'),'Mother'] = 1
    #打印母亲的存活个数
#     print full.groupby(["Survived","Mother"]).size()
    
    full['Embarked'] = pd.factorize(full['Embarked'])[0]
    full['Title'] = pd.factorize(full['Title'])[0]
    full['Sex'] = pd.factorize(full['Sex'])[0]
    full['FsizeD'] = pd.factorize(full['FsizeD'])[0]
    
    return full
if __name__ == '__main__':
    getDataSet()