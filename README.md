## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
```
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.
```

# FEATURE ENCODING:

1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
### NAME: Yenuganti Prathyusha
### REG.NO: 212223240187
```P
import pandas as pd
df=pd.read_csv('EncodingData.csv')
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
```
```P
pm=['Hot','Warm','Cold']
df
```
![image](https://github.com/user-attachments/assets/8fbd48c2-99c5-4bd8-873e-bfcef7c160de)

```P
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[['ord_2']])

```
![image](https://github.com/user-attachments/assets/ffc1f321-f178-4e98-b2d4-d7f54ce40e5c)

```P
df['bo2']=e1.fit_transform(df[['ord_2']])
df

```
![image](https://github.com/user-attachments/assets/0cb87853-f1a3-44c3-a63e-63f4f0b93716)

```P
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/d43a6e61-760a-4c0e-9f4f-5f04469ce415)

```P
from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(one.fit_transform(df2[['nom_0']]))
df2
```
![image](https://github.com/user-attachments/assets/e7018135-8974-4fc0-a6bd-b9f156e2530e)

```P
pip install --upgrade category_encoders


```
![image](https://github.com/user-attachments/assets/ca342f14-a85e-4a3e-acd9-08378ef3f80e)

```P
from category_encoders import BinaryEncoder
df=pd.read_csv('data.csv')
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/061dc2ac-20a7-43a8-a99d-6e749c1e7d57)

```P
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc['City'],y=cc['Target'])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/b193fcc0-75a4-4b97-ab72-8bd5a5fe4b45)

```P
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv('Data_to_Transform.csv')
df
```
![image](https://github.com/user-attachments/assets/866ea050-3e9c-47df-8e3e-bce0e91030a0)

```P
!pip install scikit-learn --upgrade
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/eee3590a-dc84-454c-a9e5-5949674a64d5)

```P
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/69432911-a51b-437e-91e3-0cb98302b638)

```P
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/85e98b8b-517e-4f72-925c-233935796b8c)

```P
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/b48dda39-09ae-4190-aa3b-67e3b1f874a7)

```P
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/f2908c8b-1f24-44c8-ae97-011bf3c77629)

```P
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/85bd59ef-013b-4bef-91d7-1e3a4dbf3d65)

```P
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```
![image](https://github.com/user-attachments/assets/34bac6f4-48fa-4339-ae61-4bdcd7e2a9f4)

```P
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/571840fd-14de-416a-b88c-db1f3121993f)

```P
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/ce3f5a5f-a2e5-4ad7-8dc9-076ee5dcc5e7)

```P
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/4948dbcb-a6b3-42a5-a833-62faa8c744ee)


```P
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/1119ad81-8494-4c7d-ab56-74e1ab2c5b33)


# RESULT:
Thus,using data science we performed Feature Encoding and Transformation process.

       


       
