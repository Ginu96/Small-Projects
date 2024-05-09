# %%
import uvicorn
from fastapi import FastAPI
from Cust_churn import Customer_churn
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler

# %%
scaler = MinMaxScaler()



# %%
app=FastAPI()
pkle_in=open(r"D:\Beinex\Python\Tasks\21.FastAPI_MLDeployment\dt.pkl","rb")

dt=pickle.load(pkle_in)
@app.post("/predict")
def customer_churn_pred(data:Customer_churn):
    data = data.model_dump()
    DataPlan=data["DataPlan"]
    DataUsage=data["DataUsage"]
    CustServCalls=data["CustServCalls"]
    DayMins=data["DayMins"]
    DayCalls=data["DayCalls"]
    MonthlyCharge=data["MonthlyCharge"]
    OverageFee=data["OverageFee"]
    
    #Apply the scaling of data appropriately before doing prediction
    #DayMins, DayCalls, and MonthlyCharge are scaled
    #Not sure how to validate this process
    DayMins= scaler.transform([[data[DayMins]]])[0][0]
    DayCalls = scaler.transform([[data[DayCalls]]])[0][0]
    MonthlyCharge= scaler.transform([[data[MonthlyCharge]]])[0][0]
   
    
    Y_pred= dt.predict([[DataPlan,DataUsage, CustServCalls, DayMins,
                                  DayCalls, MonthlyCharge, OverageFee]])
    if(Y_pred[0] == 0):
        print(Y_pred[0])
        pred = "Customer lost"
    else:
        pred ="Customer stays"
    print(pred)
    return {
        'prediction': pred
    }

y=dt.predict([[5,4,56,44,3,4,7]])
y[0]


# %%
#http://127.0.0.1:8000/docs

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# %%
