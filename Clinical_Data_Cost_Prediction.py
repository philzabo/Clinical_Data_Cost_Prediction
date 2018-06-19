import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn.metrics
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('Rush_Enc_Cost_DataV3.csv')
df = df.dropna()

print '\nModel training and evaluating\n'
#Transform categorical data into numerical
le_admitmonth = LabelEncoder()
le_admissionday = LabelEncoder()
le_admissionstatus = LabelEncoder()
le_sex = LabelEncoder()
le_serviceline = LabelEncoder()
le_severity = LabelEncoder()
le_mortality = LabelEncoder()
le_attending = LabelEncoder()

df['Admit Month'] = le_admitmonth.fit_transform(df['Admit Month'])
df['Admission Day'] = le_admissionday.fit_transform(df['Admission Day'])
df['Admission Status'] = le_admissionstatus.fit_transform(df['Admission Status'])
df['Sex'] = le_sex.fit_transform(df['Sex'])
df['Vizient Service Line'] = le_serviceline.fit_transform(df['Vizient Service Line'])
df['Admit Severity of Illness'] = le_severity.fit_transform(df['Admit Severity of Illness'])
df['Admit Risk of Mortality'] = le_mortality.fit_transform(df['Admit Risk of Mortality'])
df['ATTENDING'] = le_attending.fit_transform(df['ATTENDING'])

variables = ['Admit Month','Admission Day','Admission Status','Age','Sex','MSDRG Weight','Vizient Service Line', 'Admit Severity of Illness','Admit Risk of Mortality','LOS Observed','LOS Index','Num of Diag','ATTENDING']

X = df[variables]
sc = StandardScaler()
X = sc.fit_transform(X)
Y = df['DIRECT COST']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

#Train model, parameter used: 200 trees, maximum leaf nodes 5, maximum tree depth 3
regressor = ExtraTreesRegressor(n_estimators = 200, random_state=33, max_leaf_nodes=10, max_depth=5)
regressor.fit(X_train,y_train)

#prediction and evaluation
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

print 'ExtraTreesRegressor evaluating result:'
print "Train MAE: ", sklearn.metrics.mean_absolute_error(y_train, y_train_pred)
print "Train RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_train_pred))
print "Test MAE: ", sklearn.metrics.mean_absolute_error(y_test, y_test_pred)
print "Test RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred))

#################################################################################################
print '\n\nFeature importance ranking\n\n'
importances = regressor.feature_importances_
std = np.std([tree.feature_importances_ for tree in regressor.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

importance_list = []
for f in range(X.shape[1]):
    variable = variables[indices[f]]
    importance_list.append(variable)
    print("%d.%s(%f)" % (f + 1, variable, importances[indices[f]]))

# Plot importances of the data elements visually
'''
plt.figure()
plt.title("Feature importances")
plt.xticks(rotation='vertical')
plt.bar(importance_list, importances[indices],
       color="r", yerr=std[indices], align="center")
plt.show()
'''
#################################################################################################
print '\n\nPredicting on new data\n\n'

#Test Cases removed from data to be fed back into the model as a check. Comments after each line are actual costs.
FirstTestCase = ['Oct','Wednesday','Emergency',39,'Female',3.3126,'Vascular Surgery','Major','Major',1,0.19,13,77368] # $ 8930.83
SecondTestCase = ['Mar','Tuesday','Elective',81,'Male',1.6872,'Vascular Surgery','Moderate','Moderate',2,0.31,20,67775]# $ 3263.05
ThirdTestCase = ['Jul','Sunday','Urgent',60,'Male',5.3762,'Neurosurgery','Extreme','Extreme',5,0.53,23,85661]  # $ 13115.22
FirstTestCaseActual = 8930.83
SecondTestCaseActual = 3263.05
ThirdTestCaseActual = 13115.22

print 'First Test - ',str(FirstTestCase)

FirstTestCase[0] = le_admitmonth.transform([FirstTestCase[0]])[0]
FirstTestCase[1] = le_admissionday.transform([FirstTestCase[1]])[0]
FirstTestCase[2] = le_admissionstatus.transform([FirstTestCase[2]])[0]
FirstTestCase[4] = le_sex.transform([FirstTestCase[4]])[0]
FirstTestCase[6] = le_serviceline.transform([FirstTestCase[6]])[0]
FirstTestCase[7] = le_severity.transform([FirstTestCase[7]])[0]
FirstTestCase[8] = le_mortality.transform([FirstTestCase[8]])[0]
FirstTestCase[12] = le_attending.transform([FirstTestCase[12]])[0]

X = sc.transform([FirstTestCase])

cost_FirstTestCase = regressor.predict(X)[0]
print 'First Test Case Cost = $',cost_FirstTestCase, ', Difference from actual = $ ',cost_FirstTestCase-FirstTestCaseActual,'\n'

print 'Second Test - ',str(SecondTestCase)

SecondTestCase[0] = le_admitmonth.transform([SecondTestCase[0]])[0]
SecondTestCase[1] = le_admissionday.transform([SecondTestCase[1]])[0]
SecondTestCase[2] = le_admissionstatus.transform([SecondTestCase[2]])[0]
SecondTestCase[4] = le_sex.transform([SecondTestCase[4]])[0]
SecondTestCase[6] = le_serviceline.transform([SecondTestCase[6]])[0]
SecondTestCase[7] = le_severity.transform([SecondTestCase[7]])[0]
SecondTestCase[8] = le_mortality.transform([SecondTestCase[8]])[0]
SecondTestCase[12] = le_attending.transform([SecondTestCase[12]])[0]

X = sc.transform([SecondTestCase])

cost_SecondTestCase = regressor.predict(X)[0]
print 'Second Test Case Cost = $',cost_SecondTestCase,', Difference from actual = $ ',cost_SecondTestCase-SecondTestCaseActual,'\n'

print 'Third Test - ',str(ThirdTestCase)

ThirdTestCase[0] = le_admitmonth.transform([ThirdTestCase[0]])[0]
ThirdTestCase[1] = le_admissionday.transform([ThirdTestCase[1]])[0]
ThirdTestCase[2] = le_admissionstatus.transform([ThirdTestCase[2]])[0]
ThirdTestCase[4] = le_sex.transform([ThirdTestCase[4]])[0]
ThirdTestCase[6] = le_serviceline.transform([ThirdTestCase[6]])[0]
ThirdTestCase[7] = le_severity.transform([ThirdTestCase[7]])[0]
ThirdTestCase[8] = le_mortality.transform([ThirdTestCase[8]])[0]
ThirdTestCase[12] = le_attending.transform([ThirdTestCase[12]])[0]

X = sc.transform([ThirdTestCase])

cost_ThirdTestCase = regressor.predict(X)[0]
print 'Third Test Case Cost = $',cost_ThirdTestCase, 'Difference from actual = $ ',cost_ThirdTestCase-ThirdTestCaseActual,'\n'