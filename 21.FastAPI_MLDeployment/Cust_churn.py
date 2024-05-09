
# %%
from pydantic import BaseModel
# %%
class Customer_churn(BaseModel):
    DataPlan:int
    DataUsage: float
    CustServCalls:int
    DayMins:float
    DayCalls:float
    MonthlyCharge:float
    OverageFee:float
print('end')
# %%
