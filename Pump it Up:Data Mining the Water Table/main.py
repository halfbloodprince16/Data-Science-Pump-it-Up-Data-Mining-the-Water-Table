"""
Created on Thu Nov  1 00:41:27 2018

@author: hbp16
"""
import pandas as pd
import numpy as np
df = pd.read_csv('train_set_values.csv')
labeldf = pd.read_csv('train_set_labels.csv')
df['result'] = labeldf['status_group']

from sklearn.preprocessing import LabelEncoder,MinMaxScaler
le = LabelEncoder()

df['lga'] = le.fit_transform(df['lga'])
df['ward'] = le.fit_transform(df['ward'])
df['extraction_type_class'] = le.fit_transform(df['extraction_type_class'])
df['management_group'] = le.fit_transform(df['management_group'])

df['payment_type'] = le.fit_transform(df['payment_type'])
df['water_quality'] = le.fit_transform(df['water_quality'])
df['quality_group'] = le.fit_transform(df['quality_group'])

df['quantity_group'] = le.fit_transform(df['quantity_group'])
df['source_class'] = le.fit_transform(df['source_class'])
df['waterpoint_type_group'] = le.fit_transform(df['waterpoint_type_group'])

df['result'] = le.fit_transform(df['result'])

X = df.iloc[:,[1,6,7,13,14,15,16,17,26,28,30,31,32,37,39]]
Y = df.iloc[:,[40]]

from sklearn.cross_validation import train_test_split as tts
X_train,X_test,y_train,y_test = tts(X,Y,test_size=0.25,random_state=42)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10,random_state=42)
rf = rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)

testdf = pd.read_csv('test_set_values.csv')

testdf['lga'] = le.fit_transform(testdf['lga'])
testdf['ward'] = le.fit_transform(testdf['ward'])
testdf['extraction_type_class'] = le.fit_transform(testdf['extraction_type_class'])
testdf['management_group'] = le.fit_transform(testdf['management_group'])

testdf['payment_type'] = le.fit_transform(testdf['payment_type'])
testdf['water_quality'] = le.fit_transform(testdf['water_quality'])
testdf['quality_group'] = le.fit_transform(testdf['quality_group'])

testdf['quantity_group'] = le.fit_transform(testdf['quantity_group'])
testdf['source_class'] = le.fit_transform(testdf['source_class'])
testdf['waterpoint_type_group'] = le.fit_transform(testdf['waterpoint_type_group'])


testX = testdf.iloc[:,[1,6,7,13,14,15,16,17,26,28,30,31,32,37,39]]
testY = rf.predict(testX)

subdf = pd.read_csv('SubmissionFormat.csv')
subdf['status_group'] = testY    
subdf['status_group'].replace(0,'functional',inplace=True)    
subdf['status_group'].replace(1,'functional needs repair',inplace=True)
subdf['status_group'].replace(2,'non functional',inplace=True)
subdf.to_csv('sub.csv')








