import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
#import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

cardata = pd.read_csv('cars.csv')

data = cardata.drop(['Model'], axis=1)
data.describe(include='all')

data_no_rv = data.dropna(axis=0)
data_no_rv.describe(include='all')

#datasns = sns.load_dataset("data_no_rv")

#Deal with outliers

#Price
q = data_no_rv['Price'].quantile(0.99)
data_price_in = data_no_rv[data_no_rv['Price']<q]
data_price_in.describe(include='all')
sns.displot(data_no_rv['Price'])

#plt.show()

#Mileage
q = data_price_in['Mileage'].quantile(0.99)
data_mileage_in = data_price_in[data_price_in['Mileage']<q]
sns.displot(data_no_rv['Mileage'])
data_mileage_in.describe(include='all')

#plt.show()

#EngineV
q = data_price_in['EngineV'].quantile(0.99)
data_mileage_in = data_price_in[data_price_in['EngineV']<q]
sns.displot(data_mileage_in['EngineV'])
data_mileage_in.describe(include='all')

#plt.show()

#Year
q = data_price_in['Year'].quantile(0.99)
data_mileage_in = data_price_in[data_price_in['Year']<q]
sns.displot(data_mileage_in['Year'])
data_mileage_in.describe(include='all')

#plt.show()

#-----------------

#data_cleaned = data_all.reset_index(drop=True)
data_cleaned = data_mileage_in.reset_index(drop=True)
data_cleaned = data_price_in.reset_index(drop=True)
data_cleaned.describe(include='all')

cardata.describe(include='all')
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')

print(data_cleaned.columns.values)

log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)

from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]
vir = pd.DataFrame()
vir["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vir["features"] = variables.columns

data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
data_pivot = pd.pivot_table(data_no_multicollinearity, columns='Brand', values='Price')
#print(data_no_multicollinearity)
#print(data_pivot)
print(data_with_dummies)

targets = data_cleaned['log_price']
#inputs = data_cleaned.drop(['log_price'], axis=1)
#inputs = data_pivot['Brand']
inputs = data_cleaned['Price']
#inputs = data_cleaned['Price']['Year']
#print(inputs)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#scaler.fit(inputs)
#inputs_scaled = scaler.transform(inputs)
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)