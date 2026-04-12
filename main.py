# car rental price prediction
# IDC 3140 - group project

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
