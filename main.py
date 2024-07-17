from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt


datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Makine Öğrenmesi\\random forest\\aylara_gore_satis.csv")
months=datas.iloc[:,0].values.reshape(-1,1)
sales=datas.iloc[:,1].values.reshape(-1,1)
RandomFR=RandomForestRegressor(n_estimators=10,random_state=0)
RandomFR.fit(months,sales)
predict=RandomFR.predict(months)
plt.scatter(months,sales)
plt.plot(months,predict)
plt.show()












