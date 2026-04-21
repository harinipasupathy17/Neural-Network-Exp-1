<H3>ENTER YOUR NAME : HARINI P </H3>
<H3>ENTER YOUR REGISTER NO. 212224230082 </H3>
<H3>EX. NO.1</H3>
<H3>DATE:21-04-2026</H3>
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

```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv("/content/drive/MyDrive/Churn_Modelling.csv")
df

df.isnull().sum()

df.duplicated()

print(df['CreditScore'].describe())

df.info()

df.drop(['Surname','Geography','Gender'],axis=1,inplace=True)
df

Scaler=MinMaxScaler()
df1=pd.DataFrame(Scaler.fit_transform(df))
df1

X = df1.iloc[:, :-1].values
print(X)

y = df1.iloc[:,-1].values
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)

print(X_train)
print(len(X_train))

print(X_test)
print(len(X_test))
```


## OUTPUT:
<img width="1723" height="526" alt="image" src="https://github.com/user-attachments/assets/d2e73827-1af8-4e42-a33f-30bf6b4246e4" />
<img width="302" height="643" alt="image" src="https://github.com/user-attachments/assets/ebeb131c-e87e-450a-a512-376838e2b229" />
<img width="442" height="639" alt="image" src="https://github.com/user-attachments/assets/da91d46f-51a1-42dd-879b-35132e0a5fb0" />
<img width="393" height="204" alt="image" src="https://github.com/user-attachments/assets/737ffe8b-bce4-4ef3-9ee5-7418e176998e" />
<img width="534" height="471" alt="image" src="https://github.com/user-attachments/assets/0340964e-f134-4042-bc81-09b023383643" />
<img width="1393" height="523" alt="image" src="https://github.com/user-attachments/assets/b70a6925-70cc-48c8-a44f-b1f4d8ac6c3f" />
<img width="995" height="513" alt="image" src="https://github.com/user-attachments/assets/fbf4ca75-8706-4b52-be03-f680397d2d8f" />
<img width="727" height="299" alt="image" src="https://github.com/user-attachments/assets/ffdb7e39-e637-43eb-b868-704caa56245b" />
<img width="257" height="35" alt="image" src="https://github.com/user-attachments/assets/b21c1cbd-7a7e-43f3-8eaf-c4386939e34c" />
<img width="790" height="190" alt="image" src="https://github.com/user-attachments/assets/87f57dde-c56e-4f11-b43b-9f3239633055" />
<img width="812" height="190" alt="image" src="https://github.com/user-attachments/assets/fe467424-8f92-47ec-97d1-20f805fe0dc4" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
