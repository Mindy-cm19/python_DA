import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO  # 画决策树可能会用到
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

import os

os.environ['PATH'] += os.pathsep + 'D:/graphviz/bin/'
import pydotplus


# sl:satisfaction_level——False:MinMaxScaler;True:StandardScaler
# le:last_evaluation——False:MinMaxScaler;True:StandardScaler
# npr:number_project——False:MinMaxScaler;True:StandardScaler
# amh:average_monthly_hours——False:MinMaxScaler;True:StandardScaler
# tsc:time_spend_company——False:MinMaxScaler;True:StandardScaler
# wa:Work_accident——False:MinMaxScaler;True:StandardScaler
# pl5:promotion_last_5years——False:MinMaxScaler;True:StandardScaler
# dp:deparment——False:LabelEncoding;True:OneHotEncoding
# slr:salary——False:LabelEncoding;True:OneHotEncoding
# lower_d——False:NotlowerDimension
# ld_n——to n dimensions
def hr_preprocessing(sl=False, le=False, npr=False, amh=False, tsc=False, wa=False, pl5=False, dp=True, slr=False,
                     lower_d=False, ld_n=1):
    df = pd.read_csv('./data/HR.csv')
    # 1、清洗数据
    df = df.dropna(subset=['satisfaction_level', 'last_evaluation'])
    df = df[df['satisfaction_level'] <= 1][df['salary'] != 'nme']
    # 2、得到标注
    label = df['left']
    df = df.drop('left', axis=1)
    # 3、特征选择(例子中的特征先全部保留)
    # 4、特征处理
    scaler_lst = [sl, le, npr, amh, tsc, wa, pl5]
    column_lst = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours',
                  'time_spend_company', 'Work_accident', 'promotion_last_5years']
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            df[column_lst[i]] = MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            df[column_lst[i]] = StandardScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[
                0]
    scaler_lst = [dp, slr]
    column_lst = ['department', 'salary']
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            if column_lst[i] == 'salary':
                # 由于LabelEncoding会按照字母顺序来确定0,1,2，破坏了low,med,high的顺序，所以需要重新定义一个函数map_salary
                df[column_lst[i]] = [map_salary(s) for s in df[column_lst[i]].values]
            else:
                df[column_lst[i]] = LabelEncoder().fit_transform(df[column_lst[i]])
                # LabelEncoding之后，进行一下归一化处理
            df[column_lst[i]] = MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            # OneHot编码，可以直接用pandas里面的get_dummies
            df = pd.get_dummies(df, columns=[column_lst[i]])
    if lower_d:
        # PCA降维与标注Label无关，而LDA降维的n_components不能超过Label的类别（由于left只有0,1，故LDA只能降成1维）
        return PCA(n_components=ld_n).fit_transform(df.values), label
    return df, label


def map_salary(s):
    d = dict([('low', 0), ('medium', 1), ('high', 2)])
    return d.get(s, 0)  # 将low,med,high分别赋值0,1,2，如果没有找到则赋值为0


def hr_modeling(features, label):
    f_v = features.values
    f_names = features.columns.values
    l_v = label.values
    # 先把验证集分离出来，再分割训练集和测试集。训练集、验证集、测试集之比6:2:2。
    X_tt, X_validation, Y_tt, Y_validation = train_test_split(f_v, l_v, test_size=0.2)
    X_train, X_test, Y_train, Y_test = train_test_split(X_tt, Y_tt, test_size=0.25)


"""
    # 回归——分类——人工神经网络
    mdl=Sequential()
    mdl.add(Dense(50,input_dim=len(f_v[0])))  # 先建立一个稠密层表示输入层。本层的输出指定50，即为下一个隐含层的输入个数
    mdl.add(Activation('sigmoid'))  # 激活函数指定为sigmoid
    mdl.add(Dense(2))  # 该隐含层不用指定输入数（上一层的输出数已经确定为本层的输入数），输出2维。
    mdl.add(Activation('softmax'))  # 保证归一化，指定本层激活函数为softmax
    sgd=SGD(lr=0.05)  # 随机梯度下降优化器SGD，lr表示梯度下降步长
    mdl.compile(loss='mean_squared_error',optimizer='adam')  # 用adam代替sgd
    mdl.fit(X_train,np.array([[0,1] if i==1 else [1,0] for i in Y_train]),nb_epoch=5000,batch_size=8999) 
    xy_lst=[(X_train,Y_train),(X_validation,Y_validation),(X_test,Y_test)]
    d=dict([(0,'Train Set'),(1,'Validation Set'),(2,'Test Set')])
    print('-'*8,'NN','-'*8)
    f=plt.figure()
    for i in range(len(xy_lst)):
        X_part=xy_lst[i][0]
        Y_part=xy_lst[i][1]
        print(d.get(i))

#         Y_pred=mdl.predict_classes(X_part)
#         print('-ACC:',accuracy_score(Y_part,Y_pred))
#         print('-REC:',recall_score(Y_part,Y_pred))
#         print('-F-Score:',f1_score(Y_part,Y_pred))

        Y_pred=mdl.predict(X_part)
#         print(Y_pred)        
        Y_pred=np.array(Y_pred[:,1]).reshape((1,-1))[0]                
        f.add_subplot(1,3,i+1)
        fpr,tpr,threshold=roc_curve(Y_part,Y_pred)
        plt.plot(fpr,tpr)
        print('NN','AUC',auc(fpr,tpr))
        print('NN','AUC_Score',roc_auc_score(Y_part,Y_pred))
    plt.show()
"""


    models = []
#     # KNN
#     models.append(('KNN',KNeighborsClassifier(n_neighbors=3)))
#     # 朴素贝叶斯
#     models.append(('GaussianNB',GaussianNB()))
#     models.append(('BernoulliNB',BernoulliNB()))
#     # 决策树
#     models.append(('DecisionTreeGini',DecisionTreeClassifier()))
#     models.append(('DecisionTreeEntropy',DecisionTreeClassifier(criterion='entropy')))
#     # SVM
#     models.append(('SVM Classifier',SVC(C=100)))
#     # 分类——集成——随机森林
#     models.append(('RandomForest',RandomForestClassifier()))
#     # 分类——集成——Adaboost
#     models.append(('Adaboost',AdaBoostClassifier(n_estimators=100)))
#     # 回归——分类——Logistics
#     models.append(('LogisticRegression',LogisticRegression(C=1000,tol=1e-10,solver='sag',max_iter=10000)))
#     # 回归——回归树与提升树
#     models.append(('GBDT',GradientBoostingClassifier(max_depth=6,n_estimators=100)))

    for clf_name, clf in models:
        clf.fit(X_train, Y_train)
        xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]
        d = dict([(0, 'Train Set'), (1, 'Validation Set'), (2, 'Test Set')])
        print('-' * 8, clf_name, '-' * 8)
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = clf.predict(X_part)
            print(d.get(i))
            print('-ACC:', accuracy_score(Y_part, Y_pred))
            print('-REC:', recall_score(Y_part, Y_pred))
            print('-F-Score:', f1_score(Y_part, Y_pred))
            """
            # 画决策树:
            dot_data=export_graphviz(clf,out_file=None,
                                     feature_names=f_names,
                                     class_names=['NL','L'],
                                     filled=True,
                                     rounded=True,
                                     special_characters=True)
            graph=pydotplus.graph_from_dot_data(dot_data)
            graph.write_pdf('dt_tree.pdf')
            """
            """
            # 以上画决策树这一段用上StringIO可以这么写（与上段等价）：
            dot_data=StringIO()
            export_graphviz(clf,out_file=dot_data,
                            feature_names=f_names,
                            class_names=['NL','L'],
                            filled=True,
                            rounded=True,
                            special_characters=True)
            graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
            graph.write_pdf('dt_tree.pdf')
            """


def regr_test(features, label):
    #     regr=LinearRegression()
    regr = Ridge(alpha=1)  # 岭回归，alpha=0时和原来的线性回归结果一致。
    #     regr=Lasso(alpha=0.001)   # Lasso回归，alpha=0时和原来的线性回归结果一致。
    regr.fit(features.values, label.values)
    Y_pred = regr.predict(features.values)
    print('Coef:', regr.coef_)
    print('MSE:', mean_squared_error(label.values, Y_pred))
    print('RMSE:', np.sqrt(mean_squared_error(label.values, Y_pred)))
    print('MAE:', mean_absolute_error(label.values, Y_pred))
    print('R2:', r2_score(label.values, Y_pred))

def main():
    features, label = hr_preprocessing(dp=False)
    # regr_test(features[['number_project', 'average_monthly_hours']], features['last_evaluation'])
    hr_modeling(features,label)

if __name__==main():
    main()