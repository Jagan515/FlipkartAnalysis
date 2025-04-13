import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('flipkart_phones.csv')
print(df.columns)
print(df.isnull().sum())
print(df['Description'].head())
# Load CSV
df = pd.read_csv('flipkart_phones.csv')

# Convert column names to lowercase
df.columns = df.columns.str.strip().str.lower()

# Clean price (₹45,885 → 45885)
df['price'] = df['price'].str.replace(r'[^\d]', '', regex=True).astype(float)

# Convert rating to float
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Extract number of ratings and reviews
df['num_ratings'] = df['review'].str.extract(r'(\d+)\s*Ratings', flags=re.IGNORECASE)[0].astype(float)
df['num_reviews'] = df['review'].str.extract(r'(\d+)\s*Reviews', flags=re.IGNORECASE)[0].astype(float)

# Extract RAM (GB)
df['ram_gb'] = df['description'].str.extract(r'(\d+)\s*GB\s*RAM', flags=re.IGNORECASE)[0].astype(float)

# Extract Storage (GB)
df['storage_gb'] = df['description'].str.extract(r'(\d+)\s*GB\s*ROM', flags=re.IGNORECASE)[0].astype(float)

# Extract Display size
df['display_inch'] = df['description'].str.extract(r'(\d+\.\d+)\s*inch', flags=re.IGNORECASE)[0].astype(float)

# Extract Camera MP
df['camera_mp'] = df['description'].str.extract(r'(\d+)\s*MP\s*Rear', flags=re.IGNORECASE)[0].astype(float)

# Extract Battery mAh
df['battery_mah'] = df['description'].str.extract(r'(\d+)\s*mAh', flags=re.IGNORECASE)[0].astype(float)

# Extract Warranty in years
df['warranty_years'] = df['description'].str.extract(r'(\d+)\s*year', flags=re.IGNORECASE)[0].astype(float)
# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Save cleaned file
df.to_csv('checkclean.csv', index=False)

#check
print(df.columns.tolist())
#convert
df = pd.read_csv('checkclean.csv')

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Convert price to float
df['price'] = df['price'].astype(float)

# Convert rating to numeric
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Extract number of ratings and reviews
df['num_ratings'] = df['review'].str.extract(r'([\d,]+)\s*Ratings')[0]
df['num_ratings'] = df['num_ratings'].str.replace(',', '').astype(float)

df['num_reviews'] = df['review'].str.extract(r'([\d,]+)\s*Reviews')[0]
df['num_reviews'] = df['num_reviews'].str.replace(',', '').astype(float)

# Extract RAM and Storage from description
df['ram_gb'] = df['description'].str.extract(r'(\d+)\s*GB\s*RAM', flags=re.IGNORECASE)[0].astype(float)
df['storage_gb'] = df['description'].str.extract(r'(\d+)\s*GB\s*ROM', flags=re.IGNORECASE)[0].astype(float)

# Extract Display size
df['display_inch'] = df['description'].str.extract(r'(\d+\.\d+)\s*inch', flags=re.IGNORECASE)[0].astype(float)

# Extract Camera MP
df['camera_mp'] = df['description'].str.extract(r'(\d+)\s*MP', flags=re.IGNORECASE)[0]
df['camera_mp'] = pd.to_numeric(df['camera_mp'], errors='coerce')

# Extract Battery
df['battery_mah'] = df['description'].str.extract(r'(\d+)\s*mAh', flags=re.IGNORECASE)[0].astype(float)

# Extract Warranty
df['warranty_years'] = df['description'].str.extract(r'(\d+)\s*year', flags=re.IGNORECASE)[0].astype(float)

# Show preview
print(df[['product', 'ram_gb', 'storage_gb', 'display_inch', 'camera_mp', 'battery_mah', 'warranty_years']].head())

# Save intermediate cleaned data
df.to_excel("checkclean2.xlsx")

# Handling Missing Values (fixed syntax)

df['price'] = df['price'].fillna(df['price'].median())
df['rating'] = df['rating'].fillna(df['rating'].mean())
df['num_ratings'] = df['num_ratings'].fillna(0)
df['num_reviews'] = df['num_reviews'].fillna(0)
df['ram_gb'] = df['ram_gb'].fillna(df['ram_gb'].mode()[0])
df['storage_gb'] = df['storage_gb'].fillna(df['storage_gb'].mode()[0])
df['display_inch'] = df['display_inch'].fillna(df['display_inch'].median())
df['camera_mp'] = df['camera_mp'].fillna(df['camera_mp'].mode()[0])
df['battery_mah'] = df['battery_mah'].fillna(df['battery_mah'].median())
df['warranty_years'] = df['warranty_years'].fillna(1.0)

# Final cleaned data save
df.to_excel("checkclean_final.xlsx", index=False)
print("\nMissing values after cleaning:\n")
print(df.isnull().sum())

# Load your data
df = pd.read_csv('checkclean.csv')

# Fill missing values correctly
df['price'] = df['price'].fillna(df['price'].median())
df['rating'] = df['rating'].fillna(df['rating'].mean())
df['num_ratings'] = df['num_ratings'].fillna(0)
df['num_reviews'] = df['num_reviews'].fillna(0)

df['ram_gb'] = df['ram_gb'].fillna(df['ram_gb'].mode()[0])
df['storage_gb'] = df['storage_gb'].fillna(df['storage_gb'].mode()[0])

df['display_inch'] = df['display_inch'].fillna(df['display_inch'].median())
df['camera_mp'] = df['camera_mp'].fillna(df['camera_mp'].mode()[0])
df['battery_mah'] = df['battery_mah'].fillna(df['battery_mah'].median())

df['warranty_years'] = df['warranty_years'].fillna(1)

# Handle review column if needed
# Option: Drop it if already parsed
df.drop(columns=['review'], inplace=True)  # or fill with default if needed

# Optional: Save cleaned version
df.to_csv('cleaned_data.csv', index=False)
#price perdict obj 1
sns.histplot(df['price'], kde=True)
plt.title('Price Distribution')
plt.show()
#ram distribution obj 2
sns.countplot(data=df, x='ram_gb')
plt.title('RAM Distribution')
plt.show()

sns.countplot(data=df, x='storage_gb')
plt.title('Storage Distribution')
plt.show()
# correlation in colum within datset
df_numeric = df.drop(columns=[ 'product', 'description'])


plt.figure(figsize=(10, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numeric Features')
plt.show()
# unique prduct 
# Step 1: How many unique products?
print("Unique products:", df['product'].nunique())

# Step 2: Top 10 most frequent products
print("\nTop 10 most common products:")
print(df['product'].value_counts().head(10))

# Step 3: Average rating by product (Top 10)
print("\nTop 10 products by average rating:")
print(df.groupby('product')['rating'].mean().sort_values(ascending=False).head(10))
#to check the occurence of words in columns or dataset


# Combine all descriptions into a single string
text = ' '.join(df['description'].dropna().astype(str).values)

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Plot
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Common Words in Product Descriptions', fontsize=20)
plt.show()
budget_phones = df[df['price'] <= 50000]


#Distribution of dataset rating column is left squed obj 3
plt.figure(figsize=(8, 5))
sns.histplot(budget_phones['rating'].dropna(), bins=10, kde=True, color='skyblue')
plt.title('Rating Distribution for Phones Under ₹5000')
plt.xlabel('Rating')
plt.ylabel('Number of Phones')
plt.show()
#Model avaliable of brands 
# Step 1: Extract brand name (first word of product name)
# Safely create the 'brand' column
budget_phones.loc[:, 'brand'] = budget_phones['product'].str.split().str[0]


# Step 2: Count of phones by brand
plt.figure(figsize=(10, 5))
sns.countplot(data=budget_phones, x='brand', order=budget_phones['brand'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Brand Distribution for Phones Under ₹5000')
plt.xlabel('Brand')
plt.ylabel('Number of Models')
plt.show()
# part 3 of the project applying the model for analysis
features = ['ram_gb', 'storage_gb', 'display_inch', 'camera_mp', 'battery_mah', 'warranty_years']
target = 'price'

X = budget_phones[features]
y = budget_phones[target]
#making variable for linear regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
#calulation for applying linear regresssion model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
# Chart showing the spread of data
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='green')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Phone Price")
plt.grid(True)
plt.show()

#I was not satisfied with the number so..
# I thought lets check another model 


# Train a Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print(f"Random Forest MSE: {rf_mse}")
print(f"Random Forest R-squared: {rf_r2}")

#checking which columns weight in models through graph
importances = rf_model.feature_importances_
features = X.columns
plt.figure(figsize=(8, 5))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("What Influences Price the Most?")
plt.show()
# checking is error is due to null value
df.isnull().sum()
#Step 3: Extract Brand from Product Name
df['brand'] = df['product'].str.split().str[0]

#onvert it to numeric form using encoding improve the model

le = LabelEncoder()
df['brand_encoded'] = le.fit_transform(df['brand'])
# check is things are going correctly
correlation = df.corr(numeric_only=True)
print(correlation['price'].sort_values(ascending=False))


scaler = StandardScaler()
features = ['rating', 'num_ratings', 'num_reviews', 'ram_gb', 'storage_gb', 'display_inch', 'camera_mp', 'battery_mah', 'warranty_years']
df[features] = scaler.fit_transform(df[features])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Predict
y_pred_lin = lin_model.predict(X_test)

# Evaluate
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

print("📊 Linear Regression Results:")
print(f"Mean Squared Error: {mse_lin}")
print(f"R-squared: {r2_lin}")
from sklearn.ensemble import RandomForestRegressor

# Initialize and train
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n🌲 Random Forest Results:")
print(f"Mean Squared Error: {mse_rf}")
print(f"R-squared: {r2_rf}")
# Drop unnecessary columns
df.drop([ 'description', 'product'], axis=1, inplace=True)
#Agin encoding t
# Convert 'brand' to numerical using One-Hot Encoding
df = pd.get_dummies(df, columns=['brand'], drop_first=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_cols = ['rating', 'price', 'num_ratings', 'num_reviews',
                'ram_gb', 'storage_gb', 'display_inch', 'camera_mp',
                'battery_mah', 'warranty_years']

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


# Features and target
X = df.drop('price', axis=1)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluation
print("📊 Linear Regression:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

print("\n🌲 Random Forest:")
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))

#Final improved the Accuracy of model 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Set up the base model
rf = RandomForestRegressor(random_state=42)

# GridSearch
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, scoring='r2', verbose=1)

grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_

# Evaluate
y_pred = best_rf.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("🔧 Tuned Random Forest")
print("MSE:", mse)
print("R² Score:", r2)
print("Best Parameters:", grid_search.best_params_)
#improtance of column in random forest
# Get feature importances
importances = best_rf.feature_importances_
feature_names = X_train.columns

# Create DataFrame
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 12))
plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title(' Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.show()
#ram vs price distribution

plt.figure(figsize=(15, 5))
sns.boxplot(data=df, x='ram_gb', y='price')
plt.title('RAM vs Price Distribution')
plt.show()
#Dashboard for the distribution
# Sample DataFrame (replace with your real df)
df = pd.read_csv('checkclean.csv')
df['brand'] = df['product'].str.split().str[0]

# Initialize the app
app = dash.Dash(__name__)
app.title = "📱 Mobile Price Dashboard"

# Layout
app.layout = html.Div([
    html.H1("📱 Mobile Phones Analysis Dashboard", style={'textAlign': 'center'}),
    
    dcc.Dropdown(
        id='brand-filter',
        options=[{'label': brand, 'value': brand} for brand in df['brand'].unique()],
        value=df['brand'].unique()[0],
        clearable=False,
        style={'width': '50%', 'margin': 'auto'}
    ),

    html.Div([
        dcc.Graph(id='price-dist'),
        dcc.Graph(id='battery-vs-price')
    ])
])

# Callbacks
@app.callback(
    [Output('price-dist', 'figure'),
     Output('battery-vs-price', 'figure')],
    [Input('brand-filter', 'value')]
)
def update_graphs(selected_brand):
    filtered_df = df[df['brand'] == selected_brand]

    price_fig = px.histogram(filtered_df, x='price', nbins=20, title='Price Distribution')
    battery_fig = px.scatter(filtered_df, x='battery_mah', y='price', title='Battery vs Price',
                             size='rating', hover_name='product')

    return price_fig, battery_fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
