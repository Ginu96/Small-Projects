# %%
import streamlit as st 
import pandas as pd
import numpy as np
from model_predict import predict
import seaborn as sns
import matplotlib.pyplot as plt


#%%
external_link={'DT Model': 'https://en.wikipedia.org/wiki/Decision_tree#:~:text=A%20decision%20tree%20is%20a%20flowchart%2Dlike%20structure%20in%20which,taken%20after%20computing%20all%20attributes)',
               'Data Used in Study':'https://www.kaggle.com/code/abdurahmanmaarouf/customer-churn-classification'
}
# Set the background color using st.set_page_config()
st.set_page_config(layout="wide", page_title="ML Deployment", page_icon=":rocket:",initial_sidebar_state="auto")
# Define the CSS styles
styles = """
<style>
.menu-item {
    display: inline-block;
    padding: 8px 12px;
    margin: 0 4px;
    color: #ffffff;
    background-color: #ABEA7C;
    border-radius: 4px;
    text-decoration: none;
}
</style>
"""

st.write(styles, unsafe_allow_html=True)
# Create menu items as links
for label, url in external_link.items():
    st.markdown(f'<a class="menu-item" href="{url}" target="_blank">{label}</a>', unsafe_allow_html=True)

##styling app
buff1, col1, buff2,buff3 = st.sidebar.columns([1,1,1,1])
st.sidebar.markdown( "<h1 style='text-align: center; color: '#fffff';\
                    '>Beinex Training</h1>", unsafe_allow_html=True)
st.sidebar.markdown( "<h4 style='text-align: center; color: #FFFFFF;\
                    '>This is my first ML deployment using streamlit </h4>", unsafe_allow_html=True)
st.sidebar.markdown( "<h4 style='text-align: center; color: red;\
                    '>Ginu Sunny - Trainee Data in Science </h5>", unsafe_allow_html=True)

# %%
st.title("Predict whether your customer will churn or not!")
st.write('''This model will help you to find whther your telecom customer will churn or 
         not from your service using some key feature analysis of the customer data''')


# %%
st.write("Let's see how the data looks like")
data=pd.read_csv("D:\Beinex\Python\Dataset-Kaggle\churn_data.csv")
st.dataframe(data.head())
st.write("-------------------------------------------------------------------------")

# %%
#EDA
st.header("how many rows and coloumns we have in this data?")
shp=data.shape
st.write("We have ",shp[0]," rows and ",shp[1]," coloumns..looks great")
st.write("-------------------------------------------------------------------------")
st.header("Do we have any Null values in the data?")
na_val= data.isna().any().any()
st.write(na_val)

st.write("Oh we don't. Let's move........")


st.header('How these variables are distributed. Select the variable you like to check')

col1, col2 = st.columns(2)
with col1:
    st.text('dicrete variables')
    disc_var=["Churn", "ContractRenewal", "DataPlan","CustServCalls"]
    selected_variable1=st.selectbox('Select the variable', disc_var)
    chart_1=sns.countplot(x=selected_variable1,data=data)
    plt.xticks(rotation=0)
    plt.xlabel(selected_variable1)
    st.pyplot(chart_1.figure)

with col2:
    st.text('Continuous variables')
    con_var=["DataUsage","DayMins", "DayCalls", "MonthlyCharge", "OverageFee"]
    selected_variable2=st.selectbox('Select the variable', con_var)
    chart_2=sns.histplot(data[selected_variable2],kde=True)
    plt.ylabel("Density")
    plt.xlabel(selected_variable2)
    plt.title("Distribution Plot")
    st.pyplot(chart_2.figure)


st.write('''Our target variable distribution is highly inequal in distribution.''')

st.write("------------------------------------------------------------------------------------------------")

# %%
st.header('Apply the decision tree model and do the prediction')
st.write('''The independent variables used in the model to predict whether the customer will churn or not are 
         ["DataPlan", "DataUsage", "CustServCalls", "DayMins", "DayCalls", "MonthlyCharge", "OverageFee"]''')
st.text("Data splitted into 80:20 proportion for training and testing" )

var_eval=["DataPlan", "DataUsage", "CustServCalls", "DayMins", "DayCalls", "MonthlyCharge", "OverageFee"]

col1,col2,col3 = st.columns(3)
with col1:
    DataPlan = st.slider("Does the customer have active data plan?", min_value=0, max_value=1,step=1)
    CustServCalls= st.slider("How many customer service calls the customer has received?", min_value=0, max_value=10,step=1)
with col2:
    DataUsage=st.slider("Select the customer's data usage", min_value=0.00, max_value=5.00, step=0.01)
    DayMins=st.slider("Select the DayMins spend by the customer in service", min_value=0.00, max_value=500.00, step=0.1)
    DayCalls=st.slider("Select the DayCalls done by the customer ", min_value=0, max_value=300, step=1)
with col3:
    MonthlyCharge=st.slider("Select the monthly charge paid by the customer", min_value=0.00, max_value=150.00, step=0.1)
    OverageFee = st.number_input("Overage charges paid the customer", min_value=0.0, max_value=30.0,value=0.0, step=0.1)


#%%
if st.button("Predict your customer's next move"):
    result=predict(np.array([[DataPlan,DataUsage,CustServCalls,DayMins,DayCalls,MonthlyCharge,OverageFee]]))
    if result[0] == 1:
        st.text('Oops! you lost this cutomer')
    else: 
        st.text('Hurray! The customer loves your services')
    

