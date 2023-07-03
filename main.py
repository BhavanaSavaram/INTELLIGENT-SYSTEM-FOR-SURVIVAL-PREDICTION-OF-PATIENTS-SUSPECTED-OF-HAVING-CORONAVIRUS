#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler , Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import norm
from scipy import stats
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
#reading dataset
df = pd.read_csv("exp dataset.csv")
#delete columns with null values
df.drop('citizenship',axis=1,inplace=True)
df.drop('taken to the hospital',axis=1,inplace=True)
df.drop('section hospital',axis=1,inplace=True)
df.drop('contact corona disease',axis=1,inplace=True)
df.drop('sample for test',axis=1,inplace=True)
df.drop('Condition when entering the hospital',axis=1,inplace=True)
df.drop('Headache',axis=1,inplace=True)
df.drop('Dizziness',axis=1,inplace=True)
df.drop('limb plexus',axis=1,inplace=True)


# In[ ]:


df.drop('Chest Pain',axis=1,inplace=True)
df.drop('limb paresis',axis=1,inplace=True)
df.drop('Skin Lesions',axis=1,inplace=True)
df.drop('other sign',axis=1,inplace=True)
df.drop('gastrointestinal',axis=1,inplace=True)
df.drop('nausea',axis=1,inplace=True)
df.drop('vomiting',axis=1,inplace=True)
df.drop('Diarrhea',axis=1,inplace=True)
df.drop('Anorexia',axis=1,inplace=True)
df.drop('Dialysis status',axis=1,inplace=True)
df.drop('result PCR',axis=1,inplace=True)
df.drop('smell',axis=1,inplace=True)
df.drop('Taste',axis=1,inplace=True)
df.drop('Convulsions',axis=1,inplace=True)
df.drop('Smoking',axis=1,inplace=True)
df.drop('opium',axis=1,inplace=True)
df.drop('hypertension',axis=1,inplace=True)
display("NULL Values", df.isnull().sum())
display("Description",df.describe())
#store input values in X and output values in y
X= df.drop(['death'],axis=1)
y= df['death']
#graph to show features which have more impact on output variable
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# In[ ]:


#naive bayes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.naive_bayes import MultinomialNB
mb = MultinomialNB()
mb.fit(X_train, y_train)
y_pred_mb = mb.predict(X_test)
from sklearn.metrics import accuracy_score
naive=accuracy_score(y_test,y_pred_mb)
print(naive)
#logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver = 'lbfgs')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
lr=accuracy_score(y_test,y_pred)
print(lr)
#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
from sklearn.metrics import accuracy_score
k=accuracy_score(y_test,y_pred_knn)
print(k)
#SVM
from sklearn.svm import SVC
svm = SVC(kernel='linear',C=0.025, random_state=101)
svm.fit(X_train, y_train)
y_pred_svc = svm.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
s=accuracy_score(y_test,y_pred_svc)
print(s)
#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc1=RandomForestClassifier(criterion= 'gini', max_depth= 10, max_features= 'sqrt',
n_estimators= 100)
rfc1.fit(X_train, y_train)
pred=rfc1.predict(X_test)
from sklearn.metrics import accuracy_score
r=accuracy_score(y_test,pred)
print(r)
#Accuracy
yaxis=[naive,lr,k,s,r]
xaxis=['Bayes','logisticregression','knn','svm','randomforest']
plt.xlabel('model')
plt.ylabel('accuracy')
plt.bar(xaxis,yaxis)

