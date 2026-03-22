import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


from preprocess import load_and_preprocess

df=load_and_preprocess("D:/system_failure_p1/ML/dataset/data.csv")


X=df.drop('Machine failure',axis=1)
y=df['Machine failure']


#split the data into test and train state 
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)


##for logistic regression 
# lr=LogisticRegression(max_iter=1000,class_weight="balanced")
# lr.fit(X_train,y_train)

# y_pred_lf=lr.predict(X_test)
# print("IN logistic regression ")
# print(classification_report(y_test,y_pred_lf))


##for randomforest 

rf=RandomForestClassifier(    class_weight={0:1, 1:3},  # give more importance to failures
    n_estimators=200,
    max_depth=10,
    random_state=42)

rf.fit(X_train,y_train)
# y_pred_rf=rf.predict(X_test)
probs=rf.predict_proba(X_test)[:,1]
y_pred_rf=(probs >0.3).astype(int)
print("In Random forest classifier we got")
print(classification_report(y_test,y_pred_rf))
# print(y_pred_rf_thres)



import pickle 
pickle.dump(rf,open("/system_failure_p1/ML/Train/model.pkl","wb"))


