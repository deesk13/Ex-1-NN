# ENTER YOUR NAME: DEVA DHARSHINI
# ENTER YOUR REGISTER NO: 212223240026
# EX. NO.1
# DATE: 22.03.25
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
# IMPORT LIBRARIES
```

from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```
# READ THE DATASET
```
df=pd.read_csv("Churn_Modelling.csv")

### Checking Data
py
df.head()
df.tail()
df.columns
```
# CHECK THE MSSING DATA
```
df.isnull().sum()


### Check for Duplicates
py
df.duplicated()
```
# ASSIGNING Y
```
y = df.iloc[:, -1].values
print(y)
```
# CHECK FOR DUPLICATES
```
py df.duplicated()
```
# CHECK FOR OUTLIERS
```
df.describe()
```

# Dropping string values data from dataset
```
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
```
# Checking datasets after dropping string values data from dataset
```
data.head()
```
# Normalize the dataset
```
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```
# SPLIT THE DATASET
```
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
```
# TRAINING AND TESTING MODEL
```
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```
## OUTPUT:
# Data checking
![Screenshot 2025-03-24 101631](https://github.com/user-attachments/assets/aff38a50-7f6b-41dc-bd6c-6ca887ff1059)

# Missing Data
![Screenshot 2025-03-24 101720](https://github.com/user-attachments/assets/8418eb8c-a6cc-4833-9e54-0271604bcdcb)

# Duplicates identification
![Screenshot 2025-03-24 101739](https://github.com/user-attachments/assets/acb8d4cf-0008-4080-8f3c-fe28bdbff8d0)

# Vakues of 'Y'
![Screenshot 2025-03-24 101749](https://github.com/user-attachments/assets/293cc258-a007-434e-b53f-bd1f9713dfa7)

# Outliers
![Screenshot 2025-03-24 101808](https://github.com/user-attachments/assets/eb7f030e-f4a5-43eb-b34b-a73a5349d55c)

# Checking datasets after dropping string values data from dataset
![Screenshot 2025-03-24 101831](https://github.com/user-attachments/assets/c99a3309-7f66-4e97-85ee-33b91853c323)

# Normalize the dataset
![Screenshot 2025-03-24 101847](https://github.com/user-attachments/assets/c39bc4d4-f4c1-459d-a562-cc41d0c1ec94)

# Split the dataset
![Screenshot 2025-03-24 101858](https://github.com/user-attachments/assets/d597452f-dfdb-4bd6-afdd-893d317ae04c)

# Training and testing model
![Screenshot 2025-03-24 101910](https://github.com/user-attachments/assets/515ef9b7-8d9f-4dbc-bd21-d7c834e74bc6)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


