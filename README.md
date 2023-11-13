# EX-07 FEATURE SELECTION
### Aim:
To Perform the various feature selection techniques on a dataset and save the data to a file. 
### Explanation:
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

### Algorithm:
- Step1: Read the given Data
- Step2: Clean the Data Set using Data Cleaning Process.
- Step3: Apply Feature selection techniques to all the features of the data set.
- Step4: Save the data to the file.

```
Developed By: SATHISH R
Register No: 212222230138
```
### Code:
```Python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("titanic_dataset.csv")
data.head()
```
![image](https://github.com/ROHITJAIND/EX-07-FEATURE-SELECTION/assets/118707073/ae7546f8-d2cf-467e-b91e-d21e7e0976e7)
<table>
<tr>
<td>

```Python
data.isnull().sum()
sns.heatmap(data.isnull(),cbar=False)
plt.title("sns.heatmap(data.isnull(),cbar=False)")
plt.show()                 
```
</td>
<td>

<img height=99% width=49% src="https://github.com/ROHITJAIND/EX-07-FEATURE-SELECTION/assets/118707073/4ae50abe-c864-4249-a961-c729ec28673d"><img height=99% width=49% src="https://github.com/ROHITJAIND/EX-07-FEATURE-SELECTION/assets/118707073/f7566715-935d-4de6-b88f-808a3950ef8e">

</td>
</tr>
</table>

```Python
data['Age'] = data['Age'].fillna(data['Age'].dropna().median())
data['Embarked']=data['Embarked'].fillna('S')
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2
drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)
```

<table>
<tr>
<td>

```Python
sns.heatmap(data.corr(),annot=True,fmt= '.1f',ax=ax)
plt.title("HeatMap")
plt.show()
```
</td>
<td>

![download](https://github.com/ROHITJAIND/EX-07-FEATURE-SELECTION/assets/118707073/7d6e9682-d99a-4ac4-b3c0-edcadd102e12)
</td>
</tr>
</table>

<table>
<tr>
<td>

```Python
sns.heatmap(data.isnull(),cbar=False)            
plt.show()
```
</td>
<td>

![download](https://github.com/ROHITJAIND/EX-07-FEATURE-SELECTION/assets/118707073/a834455d-66f0-4986-bdd0-b67e92a34156)
</td>
</tr>
</table>

<table>
<tr>
<td>

```Python
data.Survived.value_counts(normalize=True)
        .plot(kind='bar',alpha=0.5)                  
plt.show()
```
</td>
<td>

![download](https://github.com/ROHITJAIND/EX-07-FEATURE-SELECTION/assets/118707073/01cf3e3a-2c5d-472c-8f8c-ef475da069d5)
</td>
</tr>
</table>

<table>
<tr>
<td>

```Python
data.Pclass.value_counts(normalize=True)
                .plot(kind='bar',alpha=0.5)
plt.show()
```
</td>
<td>

<img height=50% width=90% src="https://github.com/ROHITJAIND/EX-07-FEATURE-SELECTION/assets/118707073/2162c897-9b45-403f-b948-7686634ad73d">

</td>
</tr>
</table>
<table>
<tr>
<td>

```Python
plt.scatter(data.Survived,data.Age,alpha=0.1)
plt.title("Age with Survived")                                
plt.show()
```
</td>
<td>
  
![download](https://github.com/ROHITJAIND/EX-07-FEATURE-SELECTION/assets/118707073/542bcc5f-190a-4d65-8d3f-e91900648951)
</td>
</tr>
</table>

```Python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = data.drop("Survived",axis=1)
y = data["Survived"]
mdlsel = SelectKBest(chi2, k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix])
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values
#Build test and training test
X_train,X_test,y_train,y_test = train_test_split
      (features,target,test_size=0.3,random_state=42)
my_forest=RandomForestClassifier(max_depth=5,min_samples_split=10,
                n_estimators=500,random_state=5,criterion='entropy')
my_forest_ = my_forest.fit(X_train,y_train)
target_predict=my_forest_.predict(X_test)
print("Random forest score: ",accuracy_score(y_test,target_predict))
from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
print ("R2     :",r2_score(y_test,target_predict))
```
![image](https://github.com/ROHITJAIND/EX-07-FEATURE-SELECTION/assets/118707073/2cd6a5d5-a910-4025-bc95-22faccc62bc9)
### Result:
Thus, the various feature selection techniques have been performed on a given dataset successfully.
