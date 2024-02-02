# AUTHOR: Dawson VanderLugt

# Load Packages
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.io as pio
pio.templates.default = "plotly_white"

# Load Data
ames_data = pd.read_csv("data/ames_home_sales.csv")
ames_data.info()
ames_data.head()

# Scale the Data
ames_data['Sale_Price'] = ames_data['Sale_Price'] / 1000
ames_data.rename(columns={'Sale_Price': 'sale_price'}, inplace=True)

# Filter the Data
filtered_ames_data = ames_data[(ames_data['Gr_Liv_Area'] < 4000) & (ames_data['Sale_Condition'] == 'Normal')]
num_filtered_homes = filtered_ames_data.shape[0]
print(num_filtered_homes)

# Describe the supervised learning task
filtered_data = ames_data[(ames_data['Gr_Liv_Area'] < 4000) & (ames_data['Sale_Condition'] == 'Normal')]
feature_columns = ['Gr_Liv_Area', 'Year_Built']
target_column = 'sale_price'

X_train, X_test, y_train, y_test = train_test_split(filtered_data[feature_columns], filtered_data[target_column], test_size=0.2, random_state=42)
ames_train = pd.concat([X_train, y_train], axis=1)
ames_test = pd.concat([X_test, y_test], axis=1)

print(f"Training set size: {X_train.shape[0]} homes")
print(f"Test set size: {X_test.shape[0]} homes")

# Linear Regression
lr = LinearRegression().fit(X=ames_train[feature_columns], y=ames_train[target_column])
ames_train['linreg_prediction'] = lr.predict(ames_train[feature_columns])
ames_test['linreg_prediction'] = lr.predict(ames_test[feature_columns])

# Decision Tree
dt = DecisionTreeRegressor(max_depth=3)
dt.fit(X=ames_train[feature_columns], y=ames_train[target_column])
ames_train['dt_prediction'] = dt.predict(ames_train[feature_columns])
ames_test['dt_prediction'] = dt.predict(ames_test[feature_columns])

# Output results (replace with appropriate print statements as needed)
