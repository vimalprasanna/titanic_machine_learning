import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.linear_model import LogisticRegression;
from sklearn.model_selection import train_test_split;
train=pd.read_csv('titanic_train.csv');
#sns.heatmap(train.isnull(),cmap='viridis');
#plt.show();
#sns.countplot(x='Survived',data=train);
#plt.show();
#sns.countplot(x='Survived',hue='Sex',data=train);
#plt.show();
#sns.countplot(x='Survived',hue='Pclass',data=train);
#plt.show();
#sns.distplot(train['Age'].dropna());
#plt.show();
#sns.countplot(x='SibSp',data=train);
#plt.show();
train['Fare'].hist(bins=50);
plt.show();
#sns.boxplot(x='Pclass',y='Age',data=train);
#plt.show();
def imputeage(cols):
    age=cols[0];
    pclass=cols[1];
    if pd.isnull(age):
        if pclass==1:
            return 37;
        elif pclass==2:
            return 29;
        else:
            return 24;
    else:
        return age;
train['Age']=train[['Age','Pclass']].apply(imputeage,axis=1);
#sns.heatmap(train.isnull(),cmap='viridis');
#plt.show();
train.drop('Cabin',axis=1,inplace=True);
train.dropna(inplace=True);
sex=pd.get_dummies(train['Sex'],drop_first=True);
embark=pd.get_dummies(train['Embarked'],drop_first=True);
train=pd.concat([train,sex,embark],axis=1);
train.drop(['Ticket','Sex','Embarked','Name'],axis=1,inplace=True);
print(train.columns);
X=train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
       'male', 'Q', 'S']];
y=train['Survived'];
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101);
lm=LogisticRegression();
lm.fit(X_train,y_train);
predictions=lm.predict(X_test);
from sklearn.metrics import confusion_matrix;
print(confusion_matrix(y_test,predictions));




