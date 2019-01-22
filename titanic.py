#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV
import warnings
# warnings.filterwarnings('ignore')

BASE_DIR = "D:\\deep\\titanic"

# In[2]:
train = pd.read_csv(BASE_DIR + "\\input\\train.csv")
test = pd.read_csv(BASE_DIR + "\\input\\test.csv")


# In[3]:

passengerid = test.PassengerId
train.drop(['PassengerId'], axis=1, inplace=True)
test.drop(['PassengerId'], axis=1, inplace=True)


# In[4]:
train.Embarked.fillna("C", inplace=True)


# In[5]:
train.Cabin.fillna("N", inplace=True)
test.Cabin.fillna("N", inplace=True)


# In[6]:
train.Cabin = [i[0] for i in train.Cabin]
test.Cabin = [i[0] for i in test.Cabin]


# In[7]:
missing_value = test[(test.Pclass == 3) & (test.Embarked == "S") & (test.Sex == "male")].Fare.mean()
test.Fare.fillna(missing_value, inplace=True)


# In[8]:
train['Sex'] = train.Sex.apply(lambda x: 0 if x == "female" else 1)
test['Sex'] = test.Sex.apply(lambda x: 0 if x == "female" else 1)


# In[9]:
survived_summary = train.groupby("Survived")


# In[10]:
survived_summary = train.groupby("Sex")


# In[11]:
survived_summary = train.groupby("Pclass")


# In[12]:
corr = train.corr()**2


# In[13]:
male_mean = train[train['Sex'] == 1].Survived.mean()
female_mean = train[train['Sex'] == 0].Survived.mean()


# In[14]:
male = train[train['Sex'] == 1]
female = train[train['Sex'] == 0]
import random
male_sample = random.sample(list(male['Survived']),50)
female_sample = random.sample(list(female['Survived']),50)
male_sample_mean = np.mean(male_sample)
female_sample_mean = np.mean(female_sample)


# In[15]:
import scipy.stats as stats


# In[16]:
train['name_length'] = [len(i) for i in train.Name]
test['name_length'] = [len(i) for i in test.Name]
def name_length_group(size):
    a = ''
    if (size <=20):
        a = 'short'
    elif (size <=35):
        a = 'medium'
    elif (size <=45):
        a = 'good'
    else:
        a = 'long'
    return a
train['nLength_group'] = train['name_length'].map(name_length_group)
test['nLength_group'] = test['name_length'].map(name_length_group)


# In[17]:
train["title"] = [i.split('.')[0] for i in train.Name]
train["title"] = [i.split(',')[1] for i in train.title]
test["title"] = [i.split('.')[0] for i in test.Name]
test["title"]= [i.split(',')[1] for i in test.title]


# In[18]:
train["title"] = [i.replace('Ms', 'Miss') for i in train.title]
train["title"] = [i.replace('Mlle', 'Miss') for i in train.title]
train["title"] = [i.replace('Mme', 'Mrs') for i in train.title]
train["title"] = [i.replace('Dr', 'rare') for i in train.title]
train["title"] = [i.replace('Col', 'rare') for i in train.title]
train["title"] = [i.replace('Major', 'rare') for i in train.title]
train["title"] = [i.replace('Don', 'rare') for i in train.title]
train["title"] = [i.replace('Jonkheer', 'rare') for i in train.title]
train["title"] = [i.replace('Sir', 'rare') for i in train.title]
train["title"] = [i.replace('Lady', 'rare') for i in train.title]
train["title"] = [i.replace('Capt', 'rare') for i in train.title]
train["title"] = [i.replace('the Countess', 'rare') for i in train.title]
train["title"] = [i.replace('Rev', 'rare') for i in train.title]
test['title'] = [i.replace('Ms', 'Miss') for i in test.title]
test['title'] = [i.replace('Dr', 'rare') for i in test.title]
test['title'] = [i.replace('Col', 'rare') for i in test.title]
test['title'] = [i.replace('Dona', 'rare') for i in test.title]
test['title'] = [i.replace('Rev', 'rare') for i in test.title]


# In[19]:
train["has_cabin"] = [0 if i == 'N'else 1 for i in train.Cabin]
test["has_cabin"] = [0 if i == 'N'else 1 for i in test.Cabin]


# In[20]:
train['child'] = [1 if i<16 else 0 for i in train.Age]
test['child'] = [1 if i<16 else 0 for i in test.Age]


# In[21]:
train['family_size'] = train.SibSp + train.Parch+1
test['family_size'] = test.SibSp + test.Parch+1


# In[22]:
def family_group(size):
    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a


# In[23]:
train['family_group'] = train['family_size'].map(family_group)
test['family_group'] = test['family_size'].map(family_group)


# In[24]:
train['is_alone'] = [1 if i<2 else 0 for i in train.family_size]
test['is_alone'] = [1 if i<2 else 0 for i in test.family_size]


# In[25]:
train['calculated_fare'] = train.Fare/train.family_size
test['calculated_fare'] = test.Fare/test.family_size


# In[26]:
def fare_group(fare):
    a= ''
    if fare <= 4:
        a = 'Very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 20:
        a = 'mid'
    elif fare <= 45:
        a = 'high'
    else:
        a = "very_high"
    return a
train['fare_group'] = train['calculated_fare'].map(fare_group)
test['fare_group'] = test['calculated_fare'].map(fare_group)


# In[27]:
train = pd.get_dummies(train, columns=['title',"Pclass", 'Cabin','Embarked','nLength_group', 'family_group', 'fare_group'], drop_first=True)
test = pd.get_dummies(test, columns=['title',"Pclass",'Cabin','Embarked','nLength_group', 'family_group', 'fare_group'], drop_first=True)
train.drop(['Cabin_T', 'family_size','Ticket','Name', 'Fare','name_length'], axis=1, inplace=True)
test.drop(['Ticket','Name','family_size',"Fare",'name_length'], axis=1, inplace=True)


# In[28]:
train = pd.concat([train[["Survived", "Age", "Sex"]], train.loc[:,"SibSp":]], axis=1)
test = pd.concat([test[["Age", "Sex"]], test.loc[:,"SibSp":]], axis=1)


# In[29]:
from sklearn.ensemble import RandomForestRegressor
def completing_age(df):
    age_df = df.loc[:,"Age":]
    temp_train = age_df.loc[age_df.Age.notnull()] ## df with age values
    temp_test = age_df.loc[age_df.Age.isnull()] ## df without age values
    y = temp_train.Age.values ## setting target variables(age) in y
    x = temp_train.loc[:, "Sex":].values
    rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)
    rfr.fit(x, y)
    predicted_age = rfr.predict(temp_test.loc[:, "Sex":])
    df.loc[df.Age.isnull(), "Age"] = predicted_age
    return df
completing_age(train)
completing_age(test);


# In[30]:
def age_group_fun(age):
    a = ''
    if age <= 1:
        a = 'infant'
    elif age <= 4:
        a = 'toddler'
    elif age <= 13:
        a = 'child'
    elif age <= 18:
        a = 'teenager'
    elif age <= 35:
        a = 'Young_Adult'
    elif age <= 45:
        a = 'adult'
    elif age <= 55:
        a = 'middle_aged'
    elif age <= 65:
        a = 'senior_citizen'
    else:
        a = 'old'
    return a
train['age_group'] = train['Age'].map(age_group_fun)
test['age_group'] = test['Age'].map(age_group_fun)
train = pd.get_dummies(train,columns=['age_group'], drop_first=True)
test = pd.get_dummies(test,columns=['age_group'], drop_first=True);
"""train.drop('Age', axis=1, inplace=True)
test.drop('Age', axis=1, inplace=True)"""


# In[31]:
X = train.drop(['Survived'], axis=1)
y = train["Survived"]

print(X.shape)
print(y.shape)


# In[32]:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = .33, random_state = 0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



# In[33]:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
test = sc.transform(test)


# In[34]:
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix




# In[35]:
#ロジスティック回帰
# from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression()
# logreg.fit(x_train,y_train)
# y_pred = logreg.predict(x_test)
# logreg_accy = round(accuracy_score(y_pred, y_test), 3)
# print (round((logreg_accy),3))
#
# from sklearn.model_selection import cross_val_score
# cross_accuracies = cross_val_score(estimator = logreg, X = x_train, y = y_train, cv = 10, n_jobs = -1)
# logreg_cross_accy = cross_accuracies.mean()
# print (round((logreg_cross_accy),3))




# In[36]:
#ロジスティック回帰＿グリッド
# C_vals = [0.099,0.1,0.2,0.5,12,13,14,15,16,16.5,17,17.5,18]
# penalties = ['l1','l2']
# param = {'penalty': penalties,
#          'C': C_vals
#         }
# grid_search = GridSearchCV(estimator=logreg,
#                            param_grid = param,
#                            scoring = 'accuracy',
#                            cv = 10
#                           )
# grid_search = grid_search.fit(x_train, y_train)
# logreg_grid = grid_search.best_estimator_
# logreg_accy = logreg_grid.score(x_test, y_test)
# print(logreg_accy)




# In[37]:
#サポートベクトルマシン
# from sklearn.svm import SVC
# svc = SVC(kernel = 'rbf', probability=True, random_state = 1, C = 3)
# svc.fit(x_train, y_train)
# y_pred = svc.predict(x_test)
# svc_accy = round(accuracy_score(y_pred, y_test), 3)
# print(svc_accy)




# In[38]:
# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=4, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=120, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_test)
random_accy = round(accuracy_score(y_pred, y_test), 3)
print(random_accy)




# In[39]:
#ランダムフォレスト＿グリッド
# n_estimators = [50,75,100,120]
# random_state = [0,15]
# min_samples_split = [5,6,10,15,20,25]
# max_depth = [5,10,15,20,25,30]
# min_samples_leaf=[2,3,4,5,6]
# parameters = {'n_estimators':n_estimators,
# 'random_state':random_state,
# 'min_samples_split':min_samples_split,
# 'max_depth':max_depth,
# }
# randomforest_grid = GridSearchCV(randomforest,
# param_grid=parameters,
# cv=StratifiedKFold(n_splits=20, shuffle=True),
# n_jobs = -1
# )
# randomforest_grid.fit(x_train, y_train)
# #randomforest_grid.score(x_test, y_test)
# print(randomforest_grid.best_estimator_)
# randomforest_grid = randomforest_grid.best_estimator_
#
# y_pred = randomforest_grid.predict(x_test)
# randomforest_grid_accy = round(accuracy_score(y_pred, y_test), 3)
# print(randomforest_grid_accy)
#
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=10, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=4, min_samples_split=5,
#             min_weight_fraction_leaf=0.0, n_estimators=120, n_jobs=None,
#             oob_score=False, random_state=0, verbose=0, warm_start=False)
# 0.837



# In[40]:
#勾配ブ－スティング
# from sklearn.ensemble import GradientBoostingClassifier
# gradient = GradientBoostingClassifier()
# gradient.fit(x_train, y_train)
#
# y_pred = gradient.predict(x_test)
# gradient_accy = round(accuracy_score(y_pred, y_test), 3)
# print(gradient_accy)




# In[41]:
#XGブースト
# from xgboost import XGBClassifier
# XGBClassifier = XGBClassifier()
# XGBClassifier.fit(x_train, y_train)
# y_pred = XGBClassifier.predict(x_test)
# print(y_pred)
# XGBClassifier_accy = round(accuracy_score(y_pred, y_test), 3)
# print(XGBClassifier_accy)


# In[]:
#LightGBM
import lightgbm
# lgbmClassifier = lightgbm.LGBMClassifier()
# lgbmClassifier.fit(x_train, y_train)
# y_pred = lgbmClassifier.predict(x_test)
# lightgbm_accy = round(accuracy_score(y_pred, y_test), 3)

#はい全部いらないー
#インスタンスできないlgbm,fitじゃなくtrain
# dtrain = lightgbm.Dataset(x_train, y_train)
# dtest = lightgbm.Dataset(x_train, y_test, reference=dtrain)
# params = {
#         'objective': 'binary',
#         'metric': 'auc',
#         'learning_rate': 0.01,
#         'reg_lambda': 0.5,
#         'reg_alpha': 0.5,
#         'colsample_bytree': 0.8,
#         'seed': 123
#         }
# lgbmClassifier = lightgbm.train(params, dtrain)
# y_pred = lgbmClassifier.predict(x_test)
# def prob2one(prob):
#     if prob <= 0.5:
#         prob = 0
#     elif prob > 0.5:
#         prob = 1
#     return prob
# y_pred_onehot = np.array([prob2one(i) for i in y_pred])
# lightgbm_accy = round(accuracy_score(y_pred_onehot, y_test), 3)
# print(lightgbm_accy)




# In[]:
#LightGBM_grid
# n_estimators = [75,100,120]
# min_child_samples = [5,10,15,20,25]
# learning_rate=[0.1, 0.01]
# num_leaves = [29,30,31,32,33,34,35]
# parameters = {
# 'n_estimators':n_estimators,
# 'min_child_samples':min_child_samples,
# 'learning_rate':learning_rate,
# 'num_leaves':num_leaves,
# }
# lgbm_grid = GridSearchCV(lgbmClassifier,
#                         param_grid=parameters,
#                         cv=StratifiedKFold(n_splits=20, shuffle=True),
#                         n_jobs = -1
#                         )
# lgbm_grid.fit(x_train, y_train)
# print(lgbm_grid.best_estimator_)
# lgbm_grid = lgbm_grid.best_estimator_
# y_pred = lgbm_grid.predict(x_test)
# lgbm_grid_accy = round(accuracy_score(y_pred, y_test), 3)
# print(lgbm_grid_accy)

# In[]:
#LightGBM_grid_most
# lgbm_grid_most = lightgbm.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
#         importance_type='split', learning_rate=0.01, max_depth=-1,
#         min_child_samples=10, min_child_weight=0.001, min_split_gain=0.0,
#         n_estimators=75, n_jobs=-1, num_leaves=29, objective=None,
#         random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
#         subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
# lgbm_grid_most.fit(x_train, y_train)
# y_pred = lgbm_grid_most.predict(x_test)
# lgbm_grid_most_accy = round(accuracy_score(y_pred, y_test), 3)
# print(lgbm_grid_most_accy)

# In[42]:
#使うモデル
# all_models = [logreg, logreg_grid,
#               knn, knn_grid,
#               gaussian,
#               svc,
#               dectree, decisiontree_grid,
#               BaggingClassifier,
#               randomforest, randomforest_grid,
#               gradient,
#               XGBClassifier,
#               ExtraTreesClassifier,
#               ExtraTreesClassifier,
#               GaussianProcessClassifier,
#               voting_classifier,
#               ]
use_models = [randomforest,]
c = {}
for i in use_models:
    a = i.predict(x_test)
    b = accuracy_score(a, y_test)
    c[i] = b





# In[43]:
#機械学習csv
test_prediction = (max(c, key=c.get)).predict(test)
submission = pd.DataFrame({
        "PassengerId": passengerid,
        "Survived": test_prediction
    })

submission.PassengerId = submission.PassengerId.astype(int)
submission.Survived = submission.Survived.astype(int)

submission.to_csv("titanic10_submission.csv", index=False)


# In[45]:
#ディープ
# import tensorflow as tf
# from tensorflow.train import GradientDescentOptimizer
#
#
# n_inputs = 40
# n_hidden1 =20
# n_hidden2 =10
# n_outputs =2
#
# learning_rate = 0.01
#
# n_epochs = 5000
# batch_size = 50
#
# X = tf.placeholder(tf.float32, shape=(None, n_inputs),name="X")
# Y = tf.placeholder(tf.int64, shape=(None),name="Y")
#
#
# hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu)
# hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)
# logits = tf.layers.dense(hidden2, n_outputs)
#
# xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)
# loss = tf.reduce_mean(xentropy)
#
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
# training_op = optimizer.minimize(loss)
#
# correct = tf.nn.in_top_k(logits, Y, 1)
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
# init = tf.global_variables_initializer()
#
#
#
# def shuffle_batch(X, Y, batch_size):
#     rnd_idx = np.random.permutation(len(X))
#     n_batches = len(X) // batch_size
#     for batch_idx in np.array_split(rnd_idx, n_batches):
#         X_batch, y_batch = X[batch_idx], y[batch_idx]
#         yield X_batch, y_batch
#
#
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         sess.run(training_op, feed_dict={X: x_train, Y: y_train})
#         acc_batch = accuracy.eval(feed_dict={X: x_train, Y: y_train})
#         print(epoch, "Batch accuracy:", acc_batch)
#
#     train_pre = np.argmax(logits.eval(feed_dict={X:test}), axis=1)
#     #それぞれの確率を出したかったらsoftmax()
#     submission = pd.DataFrame({
#             "PassengerId": passengerid,
#             "Survived": train_pre,
#         })
#     submission.PassengerId = submission.PassengerId.astype(int)
#     submission.Survived = submission.Survived.astype(int)
    # submission.to_csv(BASE_DIR + "/" + "titanic7_submission.csv", index=False)
