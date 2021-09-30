# -*- coding: utf-8 -*-
# logistic regression
# dataset: bankchurn1.csv

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as smapi
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.feature_selection import f_classif

# read the file
path="F:/aegis/4 ml/dataset/supervised/classification/bank/churn/bankchurn1.csv"
data=pd.read_csv(path)

data.head(10)
data.shape
data.columns

# remove unwanted features from dataset
data.drop(columns=['custid','surname'],inplace=True)
data.columns

# split the columns into numeric and factors
data.dtypes
nc = data.select_dtypes(exclude='object').columns.values
fc = data.select_dtypes(include='object').columns.values

# check the distribution of the y-variable
sns.countplot(x='churn',data=data)
plt.title('Distribution of the Classes')
data.churn.value_counts()

# EDA
data.info()

# EDA on numeric data
data[nc][data[nc]==0].count()

# check for distributions / outliers / mc check

data[nc].head(20)

# convert the factors into dummy variables

data_tr = data.copy()

for c in fc:
    dummy = pd.get_dummies(data[c],drop_first=True,prefix=c)
    data_tr = data_tr.join(dummy)

# check the new columns
data_tr.columns

# remove the old factor columns
data_tr.drop(columns=fc, inplace=True)
data_tr.columns


# generic functions to split data, build model, predictions and converting prob to classes

# function: splitdata
# input: data, y, split ratio
# returns: trainx,trainy,testx,testy
def splitdata(data,y,ratio=0.3):
    trainx,testx,trainy,testy = train_test_split(data.drop(y,1),
                                                 data[y],
                                                 test_size=ratio)
    
    return(trainx,trainy,testx,testy)

# function: buildmodel
# build the logistic regression model
# input: trainx,trainy
# output: Logit model
def buildModel(trainx,trainy):
    model = smapi.Logit(trainy,trainx).fit()
    return(model)

# function: predictClass
# predict the churn and convert prob into classes
# input: predicted prob, cutoff value
def predictClass(probs,cutoff):
    if (0<=cutoff<=1):
        P = probs.copy()
        P[P < cutoff] = 0
        P[P > cutoff] = 1
        
        return(P.astype(int))

# function: cm
# prints the confusionn matrix
# input: actual Y, predicted Y
# returns: -
def cm(actual,predicted):
    # method 1
    # print(confusion_matrix(actual,predicted))
    
    # method 2: using cross tab to print confusion matrix
    df = pd.DataFrame({'actual':actual,'predicted':predicted})
    print(pd.crosstab(df.actual,df.predicted,margins=True))
    
    print("\n")
    
    # print the classification report
    print(classification_report(actual,predicted))

# function: feature selection
# returns the scores of all features of train dataset
# input: train (X and y)
def bestFeatures(trainx,trainy):
    features = trainx.columns
    
    fscore,pval = f_classif(trainx,trainy)
    
    df = pd.DataFrame({'feature':features, 'fscore':fscore,'pval':pval})
    df = df.sort_values('fscore',ascending=False)
    return(df)

# --------------------------------------------------------------------    
    
# split data
trainx1,trainy1,testx1,testy1 = splitdata(data_tr,'churn')
(trainx1.shape,trainy1.shape,testx1.shape,testy1.shape)

# build model M1
m1 = buildModel(trainx1,trainy1)

# summarise the model
m1.summary()

# predict on the test data and convert predictions into classes
p1 = m1.predict(testx1)
cutoff = 0.22
pred_y1 = predictClass(p1,cutoff)

# confusion matrix
cm(testy1,pred_y1)

# select the best features
bestFeatures(trainx1,trainy1)

# cross verify the counts using this query
# len(testy1[testy1==0])
# len(testy1[testy1==1])

data.dtypes
m1.summary2()

# data_tr1_1 = data_tr.drop('salary',1)

data.dtypes
data.active.unique()

# ----------------------------------------------------- #

# run the following command from Anaconda prompt
# pip install imbalanced-learn

from imblearn.over_sampling import SMOTE

# oversampling technique
# -----------------------
sm=SMOTE()
smX,smY = sm.fit_resample(data_tr.drop('churn',1),data_tr.churn)

# create the new dataset
data_tr2 = smX.join(smY)

# compare the 2 datasets (original / oversampled)
len(data_tr), len(data_tr2)

# compare distribution of classes (original / oversampled)
data_tr.churn.value_counts(), data_tr2.churn.value_counts()

# build and predict : Model M2
trainx2,trainy2,testx2,testy2 = splitdata(data_tr2,'churn')
trainx2.shape,trainy2.shape,testx2.shape,testy2.shape

m2 = buildModel(trainx2,trainy2)
m2.summary()
p2 = m2.predict(testx2)
cutoff = 0.5
pred_y2 = predictClass(p2,cutoff)
cm(testy2,pred_y2)

# select the best features
bestFeatures(trainx2,trainy2)

# undersampling technique
# ------------------------
from imblearn.under_sampling import NearMiss
nm = NearMiss()
nmX,nmY = nm.fit_resample(data_tr.drop('churn',1),data_tr.churn)

# create the new dataset
data_tr3 = nmX.join(nmY)

# compare the 2 datasets (original / oversampled)
len(data_tr), len(data_tr3)

# compare distribution of classes (original / oversampled)
data_tr.churn.value_counts(), data_tr3.churn.value_counts()

# build and predict : Model M3
trainx3,trainy3,testx3,testy3 = splitdata(data_tr3,'churn')
trainx3.shape,trainy3.shape,testx3.shape,testy3.shape

m3 = buildModel(trainx3,trainy3)
m3.summary()
p3 = m3.predict(testx3)
cutoff = 0.37
pred_y3 = predictClass(p3,cutoff)
cm(testy3,pred_y3)


# balanced sampling
# ------------------
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

perc=0.75
oversamp=SMOTE(sampling_strategy = perc)
undersamp=RandomUnderSampler(sampling_strategy = perc)

steps = [('o',oversamp), ('u',undersamp)]

bsX,bsY = Pipeline(steps=steps).fit_resample(data_tr.drop('churn',1),data_tr.churn)

# create the new dataset
data_tr4 = bsX.join(bsY)

# compare the 2 datasets (original / balanced sample)
len(data_tr), len(data_tr4)

# compare distribution of classes (original / oversampled)
data_tr.churn.value_counts(), data_tr4.churn.value_counts()

# build and predict : Model M4
trainx4,trainy4,testx4,testy4 = splitdata(data_tr4,'churn')
trainx4.shape,trainy4.shape,testx4.shape,testy4.shape

m4 = buildModel(trainx4,trainy4)
m4.summary()
p4 = m4.predict(testx4)
cutoff = 0.5
pred_y4 = predictClass(p4,cutoff)
cm(testy4,pred_y4)





bsX,bsY = Pipeline(steps=steps).fit_resample(data_tr.drop('churn',1),data_tr.churn)






