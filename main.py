# car rental price prediction
# IDC 3140 group project

import subprocess
subprocess.run(['pip', 'install', 'kagglehub', '-q'], check=True)

import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier


USD_RATE = 300


#download the dataset
print("downloading")
path = kagglehub.dataset_download('mrnize/car-rental')
print('saved to:', path)

csv_file = ''
for filename in os.listdir(path):
    if filename.endswith('.csv'):
        csv_file = os.path.join(path, filename)
        print('found:', filename)
#get csv file structure

df = pd.read_csv(csv_file)
print('rows:', len(df))
print('columns:', len(df.columns))

print('first 5 rows:')
print(df.head())

print('shape:', df.shape)

print('columns:')
print(df.columns.tolist())

print('data types:')
print(df.dtypes)

print('missing values:')
print(df.isnull().sum())

print('some basic info:')
print(df.describe())
print(df.head())

print('fuel types:', df['Fuel_Type'].unique())
print('body types:', df['Body_Type'].unique())
print('car brands:', df['Car_Brand'].unique())

print('min daily rate:', df['Daily_Rate_LKR'].min())
print('max daily rate:', df['Daily_Rate_LKR'].max())
print('min total:', df['Total_Amount_LKR'].min())
print('max total:', df['Total_Amount_LKR'].max())


#start removing the bad rows in data

# fill blanks with middle value

df['Engine_CC'] = df['Engine_CC'].fillna(df['Engine_CC'].median())
df['Mileage_KM'] = df['Mileage_KM'].fillna(df['Mileage_KM'].median())

df['Customer_Age'] = df['Customer_Age'].fillna(df['Customer_Age'].median())
df['Vehicle_Year'] = df['Vehicle_Year'].fillna(df['Vehicle_Year'].median())
df['Daily_Rate_LKR'] = df['Daily_Rate_LKR'].fillna(df['Daily_Rate_LKR'].median())

# remove duplicates
df = df.drop_duplicates()

# remove weird prices
df = df[df['Total_Amount_LKR'] > 0]
df = df[df['Total_Amount_LKR'] < 500000]

print('rows after cleaning:', len(df))


print('average daily rate by body type:')
body_avg = df.groupby('Body_Type')['Daily_Rate_LKR'].mean()
print(body_avg)


print('average daily rate by fuel type:')
fuel_avg = df.groupby('Fuel_Type')['Daily_Rate_LKR'].mean()
print(fuel_avg)

print('average daily rate by transmission:')
trans_avg = df.groupby('Transmission')['Daily_Rate_LKR'].mean()
print(trans_avg)

print('top 10 most rented car brands:')
print(df['Car_Brand'].value_counts().head(10))

print('rentals by customer type:')
print(df['Customer_Type'].value_counts())


#print figs 

plt.figure(figsize=(10, 5))
plt.hist(df['Daily_Rate_LKR'], bins=50, color='steelblue')
plt.title('Daily Rental Rate Distribution')
plt.xlabel('Daily Rate (LKR)')

plt.ylabel('Number of Rentals')
plt.savefig('rate_distribution.png')
plt.show()

avg = df.groupby('Body_Type')['Daily_Rate_LKR'].mean()
plt.figure(figsize=(10, 5))
avg.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Average Daily Rate by Body Type')
plt.xlabel('Body Type')

plt.ylabel('Avg Daily Rate (LKR)')
plt.xticks(rotation=45)
plt.savefig('rate_by_body_type.png')
plt.show()

top_brands = df['Car_Brand'].value_counts().head(10)
plt.figure(figsize=(10, 5))
top_brands.plot(kind='bar', color='teal', edgecolor='black')

plt.title('Top 10 Car Brands by Number of Rentals')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.savefig('top_brands.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df['Customer_Age'], df['Daily_Rate_LKR'], alpha=0.2, color='green')
plt.title('Customer Age vs Daily Rate')
plt.xlabel('Customer Age')
plt.ylabel('Daily Rate (LKR)')
plt.savefig('age_vs_rate.png')
plt.show()


cols = ['Total_Amount_LKR', 'Daily_Rate_LKR', 'Rental_Duration_Days',
        'Vehicle_Year', 'Engine_CC', 'Mileage_KM', 'Customer_Age']
corr = df[cols].corr()
plt.figure(figsize=(9, 7))

plt.imshow(corr, cmap='coolwarm')
plt.colorbar()

plt.xticks(range(len(cols)), cols, rotation=45)
plt.yticks(range(len(cols)), cols)

plt.title('Correlation Heatmap')
plt.savefig('correlation.png')
plt.show()


if 'Month' in df.columns:
    plt.figure(figsize=(10, 5))
    monthly = df['Month'].value_counts().sort_index()
    monthly.plot(kind='bar', color='purple', edgecolor='black')
    plt.title('Number of Rentals by Month')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.savefig('rentals_by_month.png')
    plt.show()

    
print(df.shape)
conn = sqlite3.connect('car_rental.db')
df.to_sql('rentals', conn, if_exists='replace', index=False)
print('saved to database')

q1 = pd.read_sql_query(
    'select Fuel_Type, ROUND(AVG(Daily_Rate_LKR), 2) as avg_rate, COUNT(*) as total from rentals GROUP BY Fuel_Type ORDER BY avg_rate DESC',
    conn
)
print('Q1 - average daily rate by fuel type:')
print(q1)

q2 = pd.read_sql_query(
    'select Car_Brand, ROUND(AVG(Daily_Rate_LKR), 2) as avg_rate from  rentals GROUP BY Car_Brand ORDER BY avg_rate DESC LIMIT 5',
    conn
)
print('Q2 - top 5 most expensive brands:')
print(q2)

q3 = pd.read_sql_query(
    'SELECT Body_Type, ROUND(AVG(Total_Amount_LKR), 2) as avg_total, COUNT(*) as count FROM rentals GROUP BY Body_Type',
    conn
)
print('Q3 - average total by body type:')
print(q3)

q4 = pd.read_sql_query(
    'select Car_Brand, Rental_Duration_Days, Daily_Rate_LKR, Total_Amount_LKR FROM rentals where Rental_Duration_Days > 10 LIMIT 10',
    conn
)
print('Q4 - some long rentals:')
print(q4)

q5 = pd.read_sql_query(
    'SELECT Customer_Type, COUNT(*) as total, ROUND(AVG(Total_Amount_LKR), 2) as avg_total FROM rentals GROUP BY Customer_Type ORDER BY total DESC',
    conn
)
print('Q5 - rentals by customer type:')
print(q5)

conn.close()

data = df.copy()
le = LabelEncoder()



#continue 

# encode text columns
cat_cols = ['Fuel_Type', 'Transmission', 'Body_Type', 'Car_Brand', 'Customer_Type', 'Insurance_Type']
for col in cat_cols:
    data[col + '_enc'] = le.fit_transform(data[col])
    print('encoded:', col)

# made this column to help the model
data['Rental_Cost_LKR'] = data['Daily_Rate_LKR'] * data['Rental_Duration_Days']

feature_cols = [
    'Rental_Cost_LKR',
    'Daily_Rate_LKR',
    'Rental_Duration_Days',
    'Vehicle_Year',
    'Engine_CC',
    'Mileage_KM',
    'Customer_Age',
    'Fuel_Type_enc',
    'Transmission_enc',
    'Body_Type_enc',
    'Car_Brand_enc',
    'Customer_Type_enc',
    'Insurance_Type_enc',
]

X = data[feature_cols]
y = data['Total_Amount_LKR']

print('X shape:', X.shape)
print('y shape:', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print('training rows:', len(X_train))
print('testing rows:', len(X_test))

model = LinearRegression()
model.fit(X_train, y_train)
print('ok')

# helper class to make predictions easier
class RentalPredictor:
    def __init__(self, trained_model, cols):
        self.model = trained_model
        self.cols = cols

    def predict(self, daily_rate, days, fuel_enc, trans_enc, body_enc, brand_enc, cust_enc, ins_enc):
        row = {
            'Rental_Cost_LKR': daily_rate * days,
            'Daily_Rate_LKR': daily_rate,
            'Rental_Duration_Days': days,
            'Vehicle_Year': 2020,
            'Engine_CC': 1500,
            'Mileage_KM': 30000,
            'Customer_Age': 35,
            'Fuel_Type_enc': fuel_enc,
            'Transmission_enc': trans_enc,
            'Body_Type_enc': body_enc,
            'Car_Brand_enc': brand_enc,
            'Customer_Type_enc': cust_enc,
            'Insurance_Type_enc': ins_enc,
        }
        return round(self.model.predict(pd.DataFrame([row])[self.cols])[0], 2)

predictor = RentalPredictor(model, feature_cols)
print('predictor ready')

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = abs(y_test - predictions).mean()

print('R2 Score:', round(r2, 4))
print('RMSE: LKR', round(rmse, 2), '/ USD', round(rmse / USD_RATE, 2))
print('MAE: LKR', round(mae, 2), '/ USD', round(mae / USD_RATE, 2))

#display fig
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.3, color='blue', s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='perfect prediction')
plt.xlabel('Actual Total Amount (LKR)')
plt.ylabel('Predicted Total Amount (LKR)')
plt.title('Actual vs Predicted  R2 = ' + str(round(r2, 4)))
plt.legend()
plt.tight_layout()
plt.savefig('predictions.png')
plt.show()

# print first 10 rows to check
print('actual vs predicted (first 10):')
print()
for i in range(10):
    actual = round(y_test.values[i], 2)
    predicted = round(predictions[i], 2)
    print('row', i+1, '| actual:', actual, '| predicted:', predicted)

# Toyota SUV, 3 days, LKR 15000 per day
# fuel=petrol(2), trans=auto(0), body=SUV(2), brand=Toyota(5), customer=tourist(2), insurance=basic(0)
sample = {
    'Rental_Cost_LKR': 45000,
    'Daily_Rate_LKR': 15000,
    'Rental_Duration_Days': 3,
    'Vehicle_Year': 2020,
    'Engine_CC': 1500,
    'Mileage_KM': 30000,
    'Customer_Age': 35,
    'Fuel_Type_enc': 2,
    'Transmission_enc': 0,
    'Body_Type_enc': 2,
    'Car_Brand_enc': 5,
    'Customer_Type_enc': 2,
    'Insurance_Type_enc': 0,
}

input_row = pd.DataFrame([sample])[feature_cols]
price = model.predict(input_row)[0]
print('toyota SUV, 3 days:', round(price, 2), 'LKR /', round(price / USD_RATE, 2), 'USD')

# budget car - suzuki hatchback 3 days
sample2 = {
    'Rental_Cost_LKR': 24000,
    'Daily_Rate_LKR': 8000,
    'Rental_Duration_Days': 3,
    'Vehicle_Year': 2015,
    'Engine_CC': 1000,
    'Mileage_KM': 80000,
    'Customer_Age': 28,
    'Fuel_Type_enc': 0,
    'Transmission_enc': 1,
    'Body_Type_enc': 0,
    'Car_Brand_enc': 4,
    'Customer_Type_enc': 1,
    'Insurance_Type_enc': 0,
}

price2 = model.predict(pd.DataFrame([sample2])[feature_cols])[0]
print('suzuki hatchback, 3 days:', round(price2, 2), 'LKR /', round(price2 / USD_RATE, 2), 'USD')

# does our model beat just guessing the average price every time?
avg_price = y_train.mean()
baseline = [avg_price] * len(y_test)
r2_base = r2_score(y_test, baseline)
rmse_base = np.sqrt(mean_squared_error(y_test, baseline))

print('baseline - always guess average:')
print('  R2:', round(r2_base, 4), ' RMSE:', round(rmse_base, 0))
print('our model:')
print('  R2:', round(r2, 4), ' RMSE:', round(rmse, 0))

# trip cost for 1 to 14 days - budget tier (8000/day)
durations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

budget_costs = []
for d in durations:
    row = {
        'Rental_Cost_LKR': 8000 * d,
        'Daily_Rate_LKR': 8000,
        'Rental_Duration_Days': d,
        'Vehicle_Year': 2020,
        'Engine_CC': 1500,
        'Mileage_KM': 30000,
        'Customer_Age': 35,
        'Fuel_Type_enc': 2,
        'Transmission_enc': 0,
        'Body_Type_enc': 3,
        'Car_Brand_enc': 3,
        'Customer_Type_enc': 1,
        'Insurance_Type_enc': 0,
    }
    cost = model.predict(pd.DataFrame([row])[feature_cols])[0]
    budget_costs.append(round(cost, 2))

mid_costs = []
for d in durations:
    row = {
        'Rental_Cost_LKR': 15000 * d,
        'Daily_Rate_LKR': 15000,
        'Rental_Duration_Days': d,
        'Vehicle_Year': 2020,
        'Engine_CC': 1500,
        'Mileage_KM': 30000,
        'Customer_Age': 35,
        'Fuel_Type_enc': 2,
        'Transmission_enc': 0,
        'Body_Type_enc': 3,
        'Car_Brand_enc': 3,
        'Customer_Type_enc': 1,
        'Insurance_Type_enc': 0,
    }
    cost = model.predict(pd.DataFrame([row])[feature_cols])[0]
    mid_costs.append(round(cost, 2))

premium_costs = []
for d in durations:
    row = {
        'Rental_Cost_LKR': 23000 * d,
        'Daily_Rate_LKR': 23000,
        'Rental_Duration_Days': d,
        'Vehicle_Year': 2020,
        'Engine_CC': 1500,
        'Mileage_KM': 30000,
        'Customer_Age': 35,
        'Fuel_Type_enc': 2,
        'Transmission_enc': 0,
        'Body_Type_enc': 3,
        'Car_Brand_enc': 3,
        'Customer_Type_enc': 1,
        'Insurance_Type_enc': 0,
    }
    cost = model.predict(pd.DataFrame([row])[feature_cols])[0]
    premium_costs.append(round(cost, 2))

plt.figure(figsize=(12, 6))
plt.plot(durations, budget_costs, marker='o', color='steelblue', label='Budget (8000/day)')
plt.plot(durations, mid_costs, marker='o', color='coral', label='Mid (15000/day)')
plt.plot(durations, premium_costs, marker='o', color='seagreen', label='Premium (23000/day)')
plt.title('Trip Cost by Duration')
plt.xlabel('Days')
plt.ylabel('Predicted Total (LKR)')
plt.xticks(durations)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('trip_cost_planner.png')
plt.show()
