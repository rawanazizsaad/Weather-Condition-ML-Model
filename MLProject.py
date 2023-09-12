#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , StandardScaler , MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
lb=LabelEncoder()
MMS=MinMaxScaler()
sc=StandardScaler()


# In[2]:


my_df=pd.read_csv('diabetes_prediction_dataset.csv')

X=my_df.iloc[:,0:8]


X=X.dropna()

print(type(X))

Y=my_df.iloc[:,-1]

Y 


# In[3]:


# labelencoding 

X['gender']=lb.fit_transform(X['gender'])

X


# In[4]:


# OneHotEncoding

feature_cols = ['smoking_history']

encoder = OneHotEncoder()

X_encoded = pd.DataFrame(encoder.fit_transform(X[feature_cols]).toarray(),
                          columns=encoder.get_feature_names(feature_cols))

X = pd.concat([X.drop(feature_cols, axis=1), X_encoded], axis=1)

X.shape


# In[5]:


# Normaliztion 

num_cols = ['age', 'bmi', 'HbA1c_level','blood_glucose_level']
X[num_cols] = sc.fit_transform(X[num_cols])


# In[6]:


# show correlation matrix 

correlation_matrix = X.corr()
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

X=np.array(X)
Y=np.array(Y)

X 
Y


# In[7]:


# split dataset

x_train , x_test , y_train , y_test = train_test_split(X,Y,train_size=0.80 , random_state=42)


# In[8]:


# Logistic Regression 

param_grid = {
    # 'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100]
}

from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=42)
grid_search = GridSearchCV(classifier, param_grid, cv=5)

grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
train_score1 = grid_search.best_score_

print("Best hyperparameters: ", best_params)
print("Logistic Regression Classifier (train score) : ", train_score1)

accuracy1=grid_search.score(x_test, y_test)

print(f"Logistic Regression Classifier (test score) : {accuracy1}")


# In[9]:


y_pred=grid_search.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
print("Logistic Regression Classifier (confusion matrix) :\n")
print(cm)

sns.heatmap(cm, annot=True,fmt='3g')
plt.show()


# In[10]:


# DT

from sklearn.tree import DecisionTreeClassifier , export_graphviz
import graphviz

dt = DecisionTreeClassifier()

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_leaf_nodes':[5,8,10],
}

grid_search = GridSearchCV(dt, param_grid, cv=5)

grid_search.fit(x_train,y_train)

train_score2 = grid_search.best_score_

print("Best parameters:", grid_search.best_params_)

print("Decision Tree Classifier (train score) : ", train_score2)

accuracy2=grid_search.score(x_test, y_test)

print("Decision Tree Classifier (test score) : ", accuracy2)


# In[11]:


y_pred=grid_search.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
print("Decision Tree Classifier (confusion matrix) :\n")
print(cm)

sns.heatmap(cm, annot=True,fmt='3g')
plt.show()


# In[12]:


# dot_data = export_graphviz(grid_search, out_file=None, 
#                            feature_names=iris.feature_names,  
#                            class_names=iris.target_names,  
#                            filled=True, rounded=True,  
#                            special_characters=True)

# graph = graphviz.Source(dot_data)
# graph.render('DecisionTree', format='png')
# graph.view()


# In[13]:


# # SVM

# from sklearn.svm import SVC
# svm = SVC()

# param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf'], 'gamma': [0.1, 1, 10]}

# grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')

# grid_search.fit(x_train,y_train)

# train_score3 = grid_search.best_score_
# best_svm = grid_search.best_estimator_

# print("Best parameters:", grid_search.best_params_)

# print("SVM Classifier (train score) : ", train_score3)

# accuracy3=grid_search.score(x_test, y_test)

# print("SVM Classifier (test score) : ", accuracy3)


# In[14]:


# y_pred=grid_search.predict(x_test)

# cm=confusion_matrix(y_test,y_pred)
# print("SVM Classifier (confusion matrix) :\n")
# print(cm)

# sns.heatmap(cm, annot=True,fmt='3g')
# plt.show()


# In[15]:


# SVM

from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1, random_state=5000)

svm.fit(x_train,y_train)

train_score3=svm.score(x_train,y_train)

print("SVM Classifier (train score) : ", train_score3)

accuracy3=svm.score(x_test, y_test)

print("SVM Classifier (test score) : ", accuracy3)


# In[16]:


y_pred=svm.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
print("SVM Classifier (confusion matrix) :\n")
print(cm)

sns.heatmap(cm, annot=True,fmt='3g')
plt.show()


# In[17]:


# Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(max_depth=5,n_estimators=10,max_features=10, random_state=42)

rfc.fit(x_train, y_train)
train_score4=rfc.score(x_train,y_train)
print("Random Forest Classifier (test score) : ",train_score4)
accuracy4 = rfc.score(x_test, y_test)
print("Random Forest Classifier (test score) : ", accuracy4)


# In[18]:


y_pred=rfc.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print("Random Forest Classifier (confusion matrix) :\n")
print(cm)

sns.heatmap(cm, annot=True,fmt='3g')
plt.show()


# In[19]:


# KNN 

from sklearn.neighbors import KNeighborsClassifier


classifier = KNeighborsClassifier(n_neighbors = 10 , metric = 'minkowski', p = 2 )
classifier.fit(x_train, y_train)

train_score5=classifier.score(x_train,y_train)
print("KNN Classifier (train score) : ",train_score5)

accuracy5 = accuracy_score(y_test,y_pred)
print("KNN Classifier (test score) : ", accuracy5)


# In[20]:


y_pred = classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print("KNN Classifier (confusion matrix) :\n")
print(cm)

sns.heatmap(cm, annot=True,fmt='3g')
plt.show()


# In[21]:


# Naive Bayes 

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train, y_train)

train_score6=classifier.score(x_train,y_train)
print("Naive Bayes Classifier (train score) : ",train_score6)

accuracy6 = accuracy_score(y_test,y_pred)
print("Naive Bayes Classifier (test score) : ", accuracy6)


# In[22]:


y_pred = nb.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
print("Naive Bayes Classifier (confusion matrix) :\n")
print(cm)

sns.heatmap(cm, annot=True,fmt='3g')
plt.show()


# In[23]:


# plot all models 

train_scores=[train_score1,train_score2,train_score3,train_score4,train_score5,train_score6]
test_scores=[accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6]

models = ['Logistic Regression', 'DecisionTree', 'SVM', 'RandomForest', 'KNN', 'GaussianNaiveBayes']

x = np.arange(len(models))

width = 0.25

fig, ax = plt.subplots(figsize=(20, 10))

rects1 = ax.bar(x - width, train_scores, width, label='Train Accuracy')

rects2 = ax.bar(x + width, test_scores, width, label='Test Accuracy')

ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Comparison of Training and Test Accuracies')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()

