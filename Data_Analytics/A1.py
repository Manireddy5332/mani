import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt
from scipy import stats

# Step 1: Load the data into a Pandas DataFrame
data = pd.read_csv('auto-mpg.csv')

# Check data types of columns and fix if necessary
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

# Step 2: Split the data into training and testing sets
X = data.drop(columns=['mpg', 'car name'])  # Features
y = data['mpg']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 3: Visualize and categorize columns
numeric_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(exclude=[np.number]).columns

# Identify columns with symmetric and skewed distributions
numeric_symmetric_cols = ['weight', 'acceleration']
numeric_skewed_cols = ['cylinders', 'displacement', 'horsepower', 'model year', 'origin']
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.hist(X_train[col], bins=20)
    plt.title(f'Histogram of {col}')
    
    plt.subplot(1, 2, 2)
    stats.probplot(X_train[col], plot=plt)
    plt.title(f'Probability Plot of {col}')
    
    plt.tight_layout()
    plt.show()
    
    skewness = stats.skew(X_train[col])
    if abs(skewness) < 0.5:
        numeric_symmetric_cols.append(col)
    else:
        numeric_skewed_cols.append(col)


# Step 4: Build the processing pipeline
numeric_symmetric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

numeric_skewed_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('log_transform', FunctionTransformer(np.log1p, validate=False)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num_sym', numeric_symmetric_transformer, numeric_symmetric_cols),
        ('num_skewed', numeric_skewed_transformer, numeric_skewed_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 5: Train and transform the pipeline on training and testing data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Step 6: Print the shape of processed data
print("Shape of processed training data:", X_train_processed.shape)
print("Shape of processed testing data:", X_test_processed.shape)
