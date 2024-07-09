import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.inspection import permutation_importance

input_data = pd.read_csv('diedai.csv')
y=input_data.iloc[:,8]
x=input_data.iloc[:,[0,1,2,3,5,6,7]]

#Normalize all data
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
x=min_max_scaler.fit_transform(x)
#Split the data into 80% train and 20% WorkFunction data sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=22)


#Extra Trees Regressor
etr=ExtraTreesRegressor(n_jobs=-1, n_estimators=500, max_features='auto')

etr.fit(x_train,y_train)
etr_y_pred=etr.predict(x_test)
y_test = np.asarray(y_test, dtype=float)
person_etr=np.corrcoef(y_test,etr_y_pred,rowvar=0)[0][1]
y_pred=DataFrame(np.array(etr_y_pred))
y_testt=DataFrame(np.array(y_test))
# pd.concat([y_testt,y_pred],axis=1).to_csv('test1.csv',header=False,index=False)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_percentage_error
mae_score = mean_absolute_error(y_test, etr_y_pred)
mse_score = mean_squared_error(y_test, etr_y_pred)
r_2_score = r2_score(y_test, etr_y_pred)
rmse = np.sqrt(mse_score)
print(mae_score,rmse,r_2_score)
plt.scatter(y_test, y_test, s=11)
plt.scatter(y_test, etr_y_pred, s=11)
plt.show()

