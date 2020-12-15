##################################BASICS###############################################
import streamlit as st
st.title("Breast Cancer Prediction APP")
#Buttons
if st.button("About Me"):
    st.text("""Hello User , My name is Ajay Goswami and I am a Intern Trainee at KVCH NOIDA of Data Science with Machine Learning Domain , i am here creating a simple application for Breast cancer analysis and  prediction using machine learning . I hope tou will like it...
    CONTACT : 9628243313
    E-mail : ajay.goswami05322gmail.com""")

#Datetime
import datetime
st.date_input("Today date",datetime.datetime.now())

st.subheader("What is Breast Cancer?")

#Audio/Video
st.video('https://www.youtube.com/watch?v=jPtCkcILCGU')

###############################################START#######################################################################

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#EDA

st.header("Exploratory Data Analysis")
st.subheader("Breast Cancer Dataset")

#Load DataFrame
cancer= load_breast_cancer()
old_columns = cancer['feature_names'].tolist() + ['target']
columnss = []
for x in old_columns:
    x= "_".join(x.split())
    columnss.append(x)


#df= pd.DataFrame(data = np.c_[ cancer['data'],cancer['target'] ],columns=cancer['feature_names'].tolist() + ['target'])
df= pd.DataFrame(data = np.c_[ cancer['data'],cancer['target'] ],columns=columnss)


#show dataset
if st.checkbox("Preview Dataset"):
    if st.button("Head"):
        st.write(df.head())
    elif st.button("Tail"):
        st.write(df.tail())
    else:
        st.write(df.head(2))

#show entire data

if st.checkbox("Show All Data"):
    st.write(df)

#show columns
if st.checkbox("Columns"):
    st.write(df.columns)

#show dimentions
data_dim =  st.radio("Dimention / Shape",("All Data","Columns","Rows"))
if data_dim == "All Data":
    st.text("Showing shape of dataset")
    st.write(df.shape)

elif data_dim == "Columns":
    st.text("showing Columns")
    st.write(df.shape[1])
else:
    st.text("rows")
    st.write(df.shape[0])

#show summary of dataset

if st.checkbox("Describe mathemetical information"):
    st.write(df.describe())

#Checking the null values
if st.checkbox("How Many Null Values"):
    st.write(df.isna().sum())

#selecting a column
col = st.selectbox("Select Column For Deatais",tuple(columnss))
st.text("showing ==>> {}...".format(col) )
st.write(df[col])
st.write("Value of -{}- Lies Between {} - {} ".format(col,df[col].min(),df[col].max()))

st.text("Graphical view of ===> {} ".format(col))
st.write(df[col].plot(kind="hist"))
st.pyplot()

st.write("Value of -{}- Lies Between {} - {} ".format(col,df[col].min(),df[col].max()))

st.subheader("Customizable Plot")
all_column_names = df.columns.tolist()
type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
selected_column_names = st.multiselect("Selected Columns To Plot",all_column_names)
if st.button("Generate Plot"):
    st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_column_names))

    if type_of_plot == 'area':
        cust_data = df [selected_column_names]
        st.area_chart(cust_data)

    elif type_of_plot == 'bar':
        cust_data = df [selected_column_names]
        st.bar_chart(cust_data)

    elif type_of_plot == 'line':
        cust_data = df [selected_column_names]
        st.line_chart(cust_data)

    #Custom Plot
    elif type_of_plot == 'box':
        cust_plot = df [selected_column_names].plot(kind=type_of_plot)
        st.write(cust_plot)
        st.pyplot()

    elif type_of_plot == 'hist':
        cust_plot = df [selected_column_names].plot(kind=type_of_plot)
        st.write(cust_plot)
        st.pyplot()


    elif type_of_plot== 'kde':

        cust_plot = df[selected_column_names].plot(kind=type_of_plot)
        st.write(cust_plot)
        st.pyplot()


#if st.checkbox("Show Correlation with Seaborn"):
#    st.write(sns.heatmap(df.corr(),annot=True))
#    st.pyplot()
st.header("Performing Machine Learning")
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train,X_test,y_train,y_test =train_test_split(X, y, random_state=42, test_size=0.2)

st.text("<<<__Enter The Values To Predict__>>>")

mean_radius_ = st.slider("mean_radius_",min_value=df["mean_radius"].min(),max_value=df["mean_radius"].max())
mean_radius_
mean_texture_ = st.slider("mean_texture_",min_value=df["mean_texture"].min(),max_value=df["mean_texture"].max())
mean_texture_
mean_perimeter_ = st.slider("mean_perimeter_",min_value=df["mean_perimeter"].min(),max_value=df["mean_perimeter"].max())
mean_perimeter_
mean_area_ = st.slider("mean_area_",min_value=df["mean_area"].min(),max_value=df["mean_area"].max())
mean_area_
mean_smoothness_ = st.slider("mean_smoothness_",min_value=df["mean_smoothness"].min(),max_value=df["mean_smoothness"].max())
mean_smoothness_
mean_compactness_ = st.slider("mean_compactness_",min_value=df["mean_compactness"].min(),max_value=df["mean_compactness"].max())
mean_compactness_
mean_concavity_= st.slider("mean_concavity_",min_value=df["mean_concavity"].min(),max_value=df["mean_concavity"].max())
mean_concavity_
mean_concave_points_ = st.slider("mean_concave_points_",min_value=df["mean_concave_points"].min(),max_value=df["mean_concave_points"].max())
mean_concave_points_

mean_symmetry_= st.slider("mean_symmetry_",min_value=df["mean_symmetry"].min(),max_value=df["mean_symmetry"].max())
mean_symmetry_
mean_fractal_dimension_ = st.slider("mean_fractal_dimension_",min_value=df["mean_fractal_dimension"].min(),max_value=df["mean_fractal_dimension"].max())
mean_fractal_dimension_
radius_error_= st.slider("radius_error_",min_value=df["radius_error"].min(),max_value=df["radius_error"].max())
radius_error_
texture_error_ = st.slider("texture_error_",min_value=df["texture_error"].min(),max_value=df["texture_error"].max())
texture_error_
perimeter_error_= st.slider("perimeter_error_",min_value=df["perimeter_error"].min(),max_value=df["perimeter_error"].max())
perimeter_error_
area_error_ = st.slider("area_error_",min_value=df["area_error"].min(),max_value=df["area_error"].max())
area_error_
smoothness_error_= st.slider("smoothness_error_",min_value=df["smoothness_error"].min(),max_value=df["smoothness_error"].max())
smoothness_error_
compactness_error_ = st.slider("compactness_error_",min_value=df["compactness_error"].min(),max_value=df["compactness_error"].max())
compactness_error_
concavity_error_= st.slider("concavity_error_",min_value=df["concavity_error"].min(),max_value=df["concavity_error"].max())
concavity_error_
concave_points_error_ = st.slider("concave_points_error_",min_value=df["concave_points_error"].min(),max_value=df["concave_points_error"].max())
concave_points_error_
symmetry_error_= st.slider("symmetry_error_",min_value=df["symmetry_error"].min(),max_value=df["symmetry_error"].max())
symmetry_error_
fractal_dimension_error_= st.slider("fractal_dimension_error_",min_value=df["fractal_dimension_error"].min(),max_value=df["fractal_dimension_error"].max())
fractal_dimension_error_
worst_radius_= st.slider("worst_radius_",min_value=df["worst_radius"].min(),max_value=df["worst_radius"].max())
worst_radius_
worst_texture_= st.slider("worst_texture_",min_value=df["worst_texture"].min(),max_value=df["worst_texture"].max())
worst_texture_
worst_perimeter_= st.slider("worst_perimeter_",min_value=df["worst_perimeter"].min(),max_value=df["worst_perimeter"].max())
worst_perimeter_
worst_area_= st.slider("worst_area_",min_value=df["worst_area"].min(),max_value=df["worst_area"].max())
worst_area_
worst_smoothness_= st.slider("worst_smoothness_",min_value=df["worst_smoothness"].min(),max_value=df["worst_smoothness"].max())
worst_smoothness_
worst_compactness_= st.slider("worst_compactness_",min_value=df["worst_compactness"].min(),max_value=df["worst_compactness"].max())
worst_compactness_
worst_concavity_= st.slider("worst_concavity_",min_value=df["worst_concavity"].min(),max_value=df["worst_concavity"].max())
worst_concavity_
worst_concave_points_= st.slider("worst_concave_points_",min_value=df["worst_concave_points"].min(),max_value=df["worst_concave_points"].max())
worst_concave_points_
worst_symmetry_= st.slider("worst_symmetry_",min_value=df["worst_symmetry"].min(),max_value=df["worst_symmetry"].max())
worst_symmetry_
worst_fractal_dimension_= st.slider("worst_fractal_dimension_",min_value=df["worst_fractal_dimension"].min(),max_value=df["worst_fractal_dimension"].max())
worst_fractal_dimension_




test_values = pd.DataFrame({"mean_radius" : [mean_radius_],
                                "mean_texture": [mean_texture_],
                                "mean_perimeter" :[mean_perimeter_],
                                "mean_area" : [mean_area_],
                                "mean_smoothness":[mean_smoothness_],
                                "mean_compactness":[mean_compactness_],
                                "mean_concavity":[mean_concavity_],
                                "mean_concave_points":[mean_concave_points_],
                                "mean_symmetry":[mean_symmetry_],
                                "mean_fractal_dimension":[mean_fractal_dimension_],
                                "radius_error":[radius_error_],
                                "texture_error":[texture_error_],
                                "perimeter_error":[perimeter_error_],
                                "area_error":[area_error_],
                                "smoothness_error":[smoothness_error_],
                                "compactness_error":[compactness_error_],
                                "concavity_error":[concavity_error_],
                                "concave_points_error":[concave_points_error_],
                                "symmetry_error":[symmetry_error_],
                                "fractal_dimension_error":[fractal_dimension_error_],
                                "worst_radius":[worst_radius_],
                                "worst_texture":[worst_texture_],
                                "worst_perimeter":[worst_perimeter_],
                                "worst_area":[worst_area_],
                                "worst_smoothness":[worst_smoothness_],
                                "worst_compactness":[worst_compactness_],
                                "worst_concavity":[worst_concavity_],
                                "worst_concave_points":[worst_concave_points_],
                                "worst_symmetry":[worst_symmetry_],
                                "worst_fractal_dimension":[worst_fractal_dimension_]
                                })

def report(pred):
    if pred == 1:
        return ("CANCER REPORT IS POSITIVE")
    else:
        return ("CANCER REPORT IS NEGATIVE")
model = st.radio("Choose a Model",("Logistic Regression","K-Nearest Neighbors","Support Vector Machine","Random Forest"))
bt = st.button("PREDICT")
if bt:
    if model == "Logistic Regression":
        LR = LogisticRegression(C=1.0,max_iter=100)
        LR.fit(X_train,y_train)
        pred = LR.predict(test_values)
        st.write(report(pred))
        y_predict = LR.predict(X_test)
        Acc = accuracy_score(y_test, y_predict)
        st.write("Accuracy of our result from model {} is - {} ".format(model,Acc))

    elif model == "K-Nearest Neighbors":
        LN = KNeighborsClassifier(n_neighbors=5,leaf_size=30)
        LN.fit(X_train,y_train)
        pred = LN.predict(test_values)
        st.write(report(pred))
        y_predict = LN.predict(X_test)
        Acc = accuracy_score(y_test, y_predict)
        st.write("Accuracy of our result from model {} is - {} ".format(model,Acc))

    elif model == "Support Vector Machine":
        SV = SVC(C=1.0,degree=3)
        SV.fit(X_train,y_train)
        pred = SV.predict(test_values)
        st.write(report(pred))
        y_predict = SV.predict(X_test)
        Acc = accuracy_score(y_test, y_predict)
        st.write("Accuracy of our result from model {} is - {} ".format(model,Acc))


    elif model == "Random Forest":
        RF = RandomForestClassifier(n_estimators=100,min_samples_split=2)
        RF.fit(X_train,y_train)
        pred = RF.predict(test_values)
        st.write(report(pred))
        y_predict = RF.predict(X_test)
        Acc = accuracy_score(y_test, y_predict)
        st.write("Accuracy of our result from model {} is - {} ".format(model,Acc))




