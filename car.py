# Importing the neccesary packages
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st

# Loading the dataset
df=pd.read_csv("C:/Users/saket/OneDrive/Desktop/dataset.csv")


# Data preprocessing 
le=LabelEncoder()
#label Encoding the data
x=le.fit_transform(df["company"])
df.drop("company",axis=1,inplace=True)
df.insert(1,"company",x)

print(df)

X=df[['company','year','kms_driven','fuel_type']]
y=df['Price']
# spliting the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# Training the data
model=LinearRegression()
model.fit(X_train,y_train)


# Predicting the output
y_pred=model.predict(X_test)

print(r2_score(y_test,y_pred))



st.title("CAR PRICE PREDICTION")
km=st.number_input("Enter kilometers driven")
year=st.number_input("Enter year")




ft=st.radio("Fuel Type ",("Petrol","Diesel"))
if ft=="Petrol":
    ft=1
elif ft=="Diesel":
    ft=0
li=[9,year,km,ft]
#print(li)
predict=model.predict([li])
if st.button("Submit"):
    st.success(predict)
