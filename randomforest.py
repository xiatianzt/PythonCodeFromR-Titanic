# -*- coding: utf-8 -*-
'''
Created on 2016��5��8��

@author: zt
'''
import loaddata

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
def predicAndWrite(bstrfc,test):
    global full
    
    predictions = bstrfc.predict(test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title',
                        'FsizeD','Child','Mother']])
    solution = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})
    solution.to_csv('rf_Solution.csv',index=False)

def anaFeaImp(bstrfc):
    global full
    featurelist = np.array(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title',
                        'FsizeD','Child','Mother'])
    fi = (bstrfc.feature_importances_/bstrfc.feature_importances_.max())*100.0
    fi_threshold = 15
    imp_index = np.where(fi>fi_threshold)[0]
    ifs = featurelist[imp_index]
    sorted_index = np.argsort(fi[imp_index])[::-1]
    pos = np.arange(sorted_index.shape[0]) + .5
    plt.subplot(1, 1, 1)
    plt.barh(pos, fi[imp_index][sorted_index[::-1]], align='center')
    plt.yticks(pos, ifs[sorted_index[::-1]])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
#     plt.draw()
#     plt.show()

def buildModel(train,test):
    global full
    
    treenum = {"n_estimators": range(1,50)}
    rfc = RandomForestClassifier(oob_score=True)
    treesearch = GridSearchCV(rfc,treenum,n_jobs=-1,cv=10)
    treesearch.fit(train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title',
                        'FsizeD','Child','Mother']], train['Survived'])
    meanscors = []
    for it in treesearch.grid_scores_:
        meanscors.append(it.mean_validation_score)
        
    #####画出score随着树个数的变化
#     plt.plot(meanscors)
#     plt.show()
    return treesearch.best_estimator_
def getDataSet():
    global full
    
    full = loaddata.getDataSet()
    

if __name__ == '__main__':
    global full,train,test
    
    ######获取数据集
    getDataSet()
    
    train = full[:891]
    test = full[891:]
    
#     print train.describe()
    ####生成模型
    bstrfc = buildModel(train,test)
    
    #####分析各个特征的重要性
    anaFeaImp(bstrfc)
    
    #####进行预测，生成结果文件
    predicAndWrite(bstrfc,test)