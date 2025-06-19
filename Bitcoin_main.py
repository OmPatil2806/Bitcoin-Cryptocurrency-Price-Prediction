# Bitcoin & Cryptocurrency Price Prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sns
import xgboost
import yfinance as yf
import ta # Technical Analysis library
# Function 1:- Data Collection 🔽,1. Download BTC, ETH data using yfinance
def download_and_save_crypto_data():
    # Download BTC data from 2020-01-01 up to today
    btc = yf.download('BTC-USD', start='2020-01-01', auto_adjust=False)

    # Download ETH data from 2020-01-01 up to today
    eth = yf.download('ETH-USD', start='2020-01-01', auto_adjust=False)

    # Focus on BTC price (Close) and volume
    btc_focus = btc[['Close', 'Volume']].copy()
    # 'BTC_Close'	Column with Bitcoin's closing prices.
    btc_focus.rename(columns={'Close': 'BTC_Close', 'Volume': 'BTC_Volume'}, inplace=True)

    # Similarly for ETH
    eth_focus = eth[['Close', 'Volume']].copy()
    eth_focus.rename(columns={'Close': 'ETH_Close', 'Volume': 'ETH_Volume'}, inplace=True)

    # Save data to CSV files
    btc_focus.to_csv('btc_focus.csv')
    eth_focus.to_csv('eth_focus.csv')

    print("✅ BTC and ETH data downloaded and saved as 'btc_focus.csv' and 'eth_focus.csv'.")

if __name__ == "__main__":
    download_and_save_crypto_data()

 # Function 2:- Feature Engineering 🛠️
 # Goal of this feature:-Enhance your dataset with technical indicators and additional features
 # 1. Capture trends (SMA, MACD) 2.Detect momentum or overbought/oversold signals (RSI) 3.Understand volatility (Bollinger Bands) 4. Optionally, use ETH price to improve predictions for BTC


def load_and_engineer_data():
    # Load BTC and ETH CSV files
    btc = pd.read_csv('btc_focus.csv', parse_dates=[0], index_col=0)
    eth = pd.read_csv('eth_focus.csv', parse_dates=[0], index_col=0)

    # Ensure all price/volume columns are numeric
    btc['BTC_Close'] = pd.to_numeric(btc['BTC_Close'], errors='coerce')
    btc['BTC_Volume'] = pd.to_numeric(btc['BTC_Volume'], errors='coerce')
    eth['ETH_Close'] = pd.to_numeric(eth['ETH_Close'], errors='coerce')
    eth['ETH_Volume'] = pd.to_numeric(eth['ETH_Volume'], errors='coerce')
    btc['MACD_Signal'] = ta.trend.macd_signal(btc['BTC_Close'])

    # Drop rows with any missing values
    btc.dropna(inplace=True)
    eth.dropna(inplace=True)

    # === Add Technical Indicators ===
    btc['SMA_14'] = ta.trend.sma_indicator(btc['BTC_Close'], window=14)
    btc['RSI_14'] = ta.momentum.rsi(btc['BTC_Close'], window=14)
    btc['MACD'] = ta.trend.macd_diff(btc['BTC_Close'])

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=btc['BTC_Close'], window=20)
    btc['BB_High'] = bb.bollinger_hband()
    btc['BB_Low'] = bb.bollinger_lband()

    # Lag Feature: Previous day's close
    btc['BTC_Close_Lag1'] = btc['BTC_Close'].shift(1)

    # Add ETH Close as correlated feature
    btc['ETH_Close'] = eth['ETH_Close']

    # Final cleanup to drop any new NaNs
    btc.dropna(inplace=True)

    # Save to new CSV
    btc.to_csv('btc_engineered.csv')
    print("\n✅ Feature engineering completed and saved as 'btc_engineered.csv'.")

    # === Terminal Output Preview ===
    print("\n📈 Sample of engineered BTC data:")
    print(btc)  # ✅ display first 5 rows

    # === Optional Plot ===
    btc[['BTC_Close', 'SMA_14']].plot(figsize=(12, 6), title='BTC Close vs SMA 14')
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.show()

# Run the function
if __name__ == "__main__":
    load_and_engineer_data()

# Function 3:- Data Cleaning 🧹 1. Drop NaNs after indicators, 2. Filter relevant columns 3.Normalize or scale features (optional)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the engineered BTC data
btc = pd.read_csv('btc_engineered.csv', parse_dates=True, index_col=0)

# 1. Handle Missing Values
btc.dropna(inplace=True)

# 2. Create Target variable: Next day's close
btc['Target_close'] = btc['BTC_Close'].shift(-1)
btc.dropna(inplace=True)

# 3. Define Features and Target
features = [
    'BTC_Close', 'BTC_Volume', 'SMA_14', 'RSI_14', 'MACD',
    'MACD_Signal', 'BB_High', 'BB_Low', 'BTC_Close_Lag1', 'ETH_Close'
]
target = 'Target_close'

x = btc[features]
y = btc[target]

# 4. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# 5. Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# ✅ Show all columns in terminal output
pd.set_option('display.max_columns', None)

# 6. Print Cleaned Sample with all columns
print("\n✅ Cleaned BTC Data (Sample with All Columns):")
print(btc.head())

# Optional: Save cleaned sets for modeling
pd.DataFrame(X_train, columns=features).to_csv('btc_X_train.csv', index=False)
pd.DataFrame(X_test, columns=features).to_csv('btc_X_test.csv', index=False)
y_train.to_csv('btc_y_train.csv')
y_test.to_csv('btc_y_test.csv')

print("\n✅ Data cleaning and preprocessing complete. Files saved:")
print("- btc_X_train.csv")
print("- btc_X_test.csv")
print("- btc_y_train.csv")
print("- btc_y_test.csv")

# Feature 4:- Machine Learning Models 🤖
# Machine learning cocepts will include 1. Train-test split 2. Scaling 3. Model training: Linear Regression, XGBoost, 4. Evaluation: MAE, RMSE, R²
# step 1:- Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load features and target data
X_train = pd.read_csv('btc_X_train.csv').values
X_test = pd.read_csv('btc_X_test.csv').values

y_train_df = pd.read_csv('btc_y_train.csv')
y_test_df = pd.read_csv('btc_y_test.csv')

# Extract the last column as target (price)
y_train = y_train_df.iloc[:, -1].values.ravel()
y_test = y_test_df.iloc[:, -1].values.ravel()

# Print samples to terminal
print("\n📊 Sample X_train:")
print(pd.read_csv('btc_X_train.csv').head())

print("\n🎯 Sample y_train:")
print(y_train_df.head())

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Evaluate Linear Regression model
print("\nLinear Regression Model Performance:")
print("MAE:", mean_absolute_error(y_test, lr_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_preds)))
print("R²:", r2_score(y_test, lr_preds))

# Train XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Evaluate XGBoost model
print("\nXGBoost Model Performance:")
print("MAE:", mean_absolute_error(y_test, xgb_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, xgb_preds)))
print("R²:", r2_score(y_test, xgb_preds))

# Plot Actual vs Predicted BTC prices
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Actual Price', linestyle='--', color='black')
plt.plot(lr_preds, label='Linear Regression Predictions', color='blue')
plt.plot(xgb_preds, label='XGBoost Predictions', color='green')
plt.title("BTC Price Prediction: Actual vs Predicted")  # Emoji removed to avoid font warning
plt.xlabel("Time Step")
plt.ylabel("BTC Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Cleaned Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Load the engineered dataset
btc = pd.read_csv('btc_engineered.csv', parse_dates=['Price'], index_col='Price')

# Step 2: Drop rows with any missing values
btc.dropna(inplace=True)

# Step 3: Create the target variable (next day close price)
btc['Target_close'] = btc['BTC_Close'].shift(-1)

# Step 4: Drop the last row (it has NaN in target)
btc.dropna(inplace=True)

# OPTIONAL: Save full cleaned features (before splitting)
btc.to_csv('btc_features_full.csv')

#print("\n✅ Cleaned data saved to 'btc_features_full.csv'")
#print(btc.head())
#print("\nColumns in cleaned dataset:", btc.columns.tolist())


# Function 5 :- We will add some more visualizations.
# Step 1:-Import Required Libraries.
# Step 2:- Load the full cleaned dataset.

df = pd.read_csv('btc_features_full.csv')
#print(df.head(20))

# Step 3:- Visualization 1- Correlation Heatmap, You're creating a correlation heatmap to visualize how different features (columns) in your DataFrame relate to each other, especially in relation to BTC price.

# You're creating a correlation heatmap to visualize how different features (columns) in your DataFrame relate to each other, especially in relation to BTC price.
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Filter only numeric columns for correlation
numeric_df = df.select_dtypes(include='number')

# ✅ Plot heatmap with correlation of only numeric data
plt.figure(figsize=(10, 6)) # This creates a new figure (i.e., plot space) for the heatmap, figsize=(10, 6) sets the size of the figure: 10 inches wide, 6 inches tall.
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f') # df.corr() = Calculate the correlation and coefficents between all neumeric column
# annot=True= Shows the actual correlation values on the heatmap,  cmap='coolwarm' = cool blue= negative correlation and warm red= Positive correlation, fmt='.2f'= Formats the numbers shown on the heatmap to 2 decimal places
plt.title('Feature Correlation Heatmap') # Adds a title to the heatmap for clarify
plt.show()

# Step 4:- Visualization 2 - Bar Chart of Average Volume by year
# Goal of this Visualization:- To visualize how the average Bitcoin trading volume changed year-by-year.
# Step 1: Ensure the date column is properly named and converted
df['Price'] = pd.to_datetime(df['Price'], errors='coerce')  # Convert to datetime safely

# Step 2: Drop rows where 'Price' couldn't be converted (if any)
df = df.dropna(subset=['Price'])

# Step 3: Extract year from the datetime
df['Year'] = df['Price'].dt.year

# Step 4: Group by year and calculate average BTC volume
avg_volume_per_year = df.groupby('Year')['BTC_Volume'].mean()

# Step 5: Plotting
avg_volume_per_year.plot(kind='bar', color='orange', figsize=(10, 5))
plt.title('Average BTC Volume per Year')
plt.ylabel('Volume')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame 'df' is already loaded

# Define correct columns from your dataset
price_col = 'BTC_Close'
volume_col = 'BTC_Volume'

# Drop rows with missing values in those columns (just in case)
df_clean = df[[price_col, volume_col]].dropna()

# Plotting the scatter plot
# 1. Each dot represents one day (or one row) in your dataset.
# 2. the dot's position shows 1. how high the BTC price was on that day (x-axis) 2. How much BTC was traded on that day (y-axis)
# 3. Example:- Imagine a dot at:- x = 30,000 (BTC price) and y = 50,000 (BTC Volume) That means = On that day BTC was priced at $30,000 and 50,000 BTC were traded.
plt.figure(figsize=(8, 5))
plt.scatter(df_clean[price_col], df_clean[volume_col], alpha=0.5, color='green')
plt.title('BTC Price vs BTC Volume')
plt.xlabel('BTC Price (Close)')
plt.ylabel('BTC Volume')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6:- Visualization 3 – Pie Chart of Bullish vs Bearish Days
# We will compare how many times BTC closed and higher vs lower than previous day.
df['Daily_change'] = df['BTC_Close'].diff()
bullish_days = (df['Daily_change'] >0).sum()
bearish_days = (df['Daily_change'] <= 0).sum()
labels = ['Bullish Days', 'Bearish Days']
sizes = [bullish_days, bearish_days]
colors = ['green', 'red']
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Bullish vs Bearish Days')
plt.axis('equal')
plt.show()

# Step 7:- Visualization 4 – Distribution of Closing Prices
plt.figure(figsize=(10, 5))
sns.histplot(df['BTC_Close'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of BTC Closing Prices')
plt.xlabel('BTC Closing Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Step 8:- Save Visualization in PNG file
# plt.savefig('btc_price_distribution.png')

# To confirm it if you want to inscept the cleaned data visually you can run
#df = pd.read_csv('btc_features_full.csv')
#print(df.head())
#print(df.info())

# Function to Display Sorted Data and Column Info
import pandas as pd

df = pd.read_csv('btc_features_full.csv')  # <- load your data here

def display_head_tail(df, sort_by='Date'):
    # Ensure all columns are shown in output
    pd.set_option('display.max_columns', None)   # Show all columns
    pd.set_option('display.width', 1000)         # Wider display
    pd.set_option('display.max_colwidth', None)  # Don't truncate column content

    print("To Display Sorted Data and Column Information:\n")

    # Display number of columns
    print(f"Total Columns: {len(df.columns)}")

    # Display column names
    print("Column Names:")
    print(df.columns.tolist())
    print("-" * 80)

    # Convert sort_by column to datetime if it’s 'Date'
    if sort_by.lower() == 'date' and sort_by in df.columns:
        try:
            df[sort_by] = pd.to_datetime(df[sort_by])
        except Exception as e:
            print(f"Warning: Couldn't convert '{sort_by}' to datetime. Sorting as-is. Error: {e}")

    # Sort if column exists
    if sort_by in df.columns:
        df_sorted = df.sort_values(by=sort_by)
    else:
        print(f"Warning: Column '{sort_by}' not found. Showing unsorted data.")
        df_sorted = df

    # Display first 20 rows
    print("\n▶ First 20 Rows:\n")
    print(df_sorted.iloc[:20])

    # Display last 20 rows
    print("\n▶ Last 20 Rows:\n")
    print(df_sorted.iloc[-20:])

# Now call the function with the actual dataframe
display_head_tail(df)

# BTC Price Prediction (Future Forecast)
# Step 1:- Load your Data & Features:-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv('btc_features_full.csv', index_col=0, parse_dates=True)

print("Columns in dataset:", df.columns.tolist())

target_col = 'BTC_Close'

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset!")

# Step 2: Prepare features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Step 3: Train-test split without shuffling (time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Train RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'btc_model.pkl')

# Step 5: Load the model (simulate fresh start)
model = joblib.load('btc_model.pkl')

# Step 6: Rolling prediction for next 7 days with feature updates

last_features = X.iloc[-1:].copy()  # last known features row
future_dates = pd.date_range(start=X.index[-1] + pd.Timedelta(days=1), periods=7)

future_close = []

for i, date in enumerate(future_dates):
    pred_close = model.predict(last_features)[0]
    future_close.append(pred_close)

    # Update lag features (example for 'BTC_Close_Lag1')
    # You must adapt this for your actual lagged features in dataset
    if 'BTC_Close_Lag1' in last_features.columns:
        last_features['BTC_Close_Lag1'] = pred_close

    # Update other lag or rolling features here as needed

# Step 7: Print predictions to terminal
print("\n📅 Bitcoin Close Price Forecast (Next 7 Days):")
for date, price in zip(future_dates, future_close):
    print(f"{date.date()} → Close: ${price:.2f}")

# Step 8: Pie chart with + / - inside labels

def format_label(date_str, curr_price, prev_price=None):
    if prev_price is None:
        sign = ""
    else:
        sign = " + " if curr_price > prev_price else " - "
    return f"{date_str}{sign}${curr_price:.2f}"

labels = []
for i, date in enumerate(future_dates):
    if i == 0:
        labels.append(format_label(date.strftime('%m-%d'), future_close[i]))
    else:
        labels.append(format_label(date.strftime('%m-%d'), future_close[i], future_close[i-1]))

plt.figure(figsize=(8,8))
plt.pie(future_close, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Predicted BTC Close Price Distribution Over Next 7 Days')
plt.show()
