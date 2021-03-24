import pandas as pd
import numpy as np
import streamlit as st
import joblib
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler =StandardScaler()



df = pd.read_csv("data.csv.xls")
df1 = df.drop("MEDV",1)

model = joblib.load("estate.pkl")

st.header("REAL ESTATE DATASET: PRICE PREDICTOR")
st.subheader("Insert the appropriate values using the input buttons by the left.")
st.text("""Attributes Information:
       1. CRIM per capita crime rate by town
       2. ZN proportion of residential land zoned for lots over 25,000 sq.ft.
       3. INDUS proportion of non-retail business acres per town
       4. CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
       5. NOX nitric oxides concentration (parts per 10 million)
       6. RM average number of rooms per dwelling
       7. AGE proportion of owner-occupied units built prior to 1940
       8. DIS weighted distances to five Boston employment centres
       9. RAD index of accessibility to radial highways
       10. TAX full-value property-tax rate per $10,000
       11. PTRATIO pupil-teacher ratio by town
       12. B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
       13. LSTAT percentage lower status of the population 
       14. MEDV Median value of owner-occupied homes in $1000's    

""")
try: 
    def user_input():
        st.sidebar.header("FEATURES")
        CRIM = st.sidebar.number_input("insert CRIM")
        ZN = st.sidebar.number_input("insert ZN")
        INDUS = st.sidebar.number_input("insert INDUS")
        CHAS = st.sidebar.number_input("insert CHAS")
        NOX = st.sidebar.number_input("insert NOX")
        RM = st.sidebar.number_input("insert RM")
        AGE = st.sidebar.number_input("insert AGE")
        DIS = st.sidebar.number_input("insert DIS")
        RAD = st.sidebar.number_input("insert RAD")
        TAX = st.sidebar.number_input("insert TAX")
        PTRATIO = st.sidebar.number_input("insert PTRATIO")
        B = st.sidebar.number_input("insert B")
        LSTAT = st.sidebar.number_input("insert LSTAT")


        
        features = pd.DataFrame({"CRIM": CRIM, "ZN":ZN, "INDUS":INDUS,"CHAS":CHAS,
                        "NOX": NOX, "RM": RM, "AGE":AGE, "DIS":DIS,
                        "RAD": RAD, "TAX":TAX, "PTRATIO": PTRATIO,
                        "B":B, "LSTAT": LSTAT}, index = [0])
        return features


    data = user_input()


    def scaling():
        st.subheader(body = "User input")
        st.dataframe(data)
        

        df2 = pd.concat([data,df1], axis = 0)  # i had to concat with the original dataset so i could perform both boxcox and scaling

        skew_feat = df2.drop(["ZN","CHAS"], 1)
        unskew_feat = df2[["ZN","CHAS"]]

        for i in skew_feat.columns:
            if skew_feat[i].skew() <-1 or skew_feat[i].skew() >1:
                skew_feat[i] = stats.boxcox(skew_feat[i])[0]

        df3 = pd.concat([skew_feat, unskew_feat], 1)

        tran = pd.DataFrame(scaler.fit_transform(df3), columns = data.columns)
        df_ = tran[:len(data)]

        st.subheader(body = "Scaled User input")
        st.dataframe(df_)

        return df_



    df_ = scaling()
    prediction = model.predict(df_)
    st.subheader(body = "Prediction")
    predict = pd.DataFrame({"MEDV": prediction})
    st.dataframe(predict)



except ValueError:
    st.header("Enter all appropriate values and ensure to click enter at the end")