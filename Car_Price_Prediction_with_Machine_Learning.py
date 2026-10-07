# ============================================================
# TASK 3: CAR PRICE PREDICTION USING MACHINE LEARNING
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# STEP 1: LOAD DATASET
# ------------------------------------------------------------

file_name = "car_data.csv"

try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print("ERROR: Dataset file not found!")
    print("Make sure 'car_data.csv' is in the same folder.")
    exit()


print("=" * 60)
print("             CAR PRICE PREDICTION")
print("=" * 60)


# ------------------------------------------------------------
# STEP 2: DISPLAY DATA
# ------------------------------------------------------------

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# ------------------------------------------------------------
# STEP 3: CLEAN COLUMN NAMES
# ------------------------------------------------------------

df.columns = df.columns.str.strip()

# Remove unnecessary spaces from text columns
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()


# ------------------------------------------------------------
# STEP 4: HANDLE MISSING VALUES
# ------------------------------------------------------------

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove rows containing missing values
df = df.dropna()

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ------------------------------------------------------------
# STEP 5: REMOVE DUPLICATES
# ------------------------------------------------------------

duplicates = df.duplicated().sum()

print("\nDuplicate rows:", duplicates)

df = df.drop_duplicates()


# ------------------------------------------------------------
# STEP 6: DISPLAY STATISTICAL INFORMATION
# ------------------------------------------------------------

print("\nStatistical Summary:")
print(df.describe())


# ------------------------------------------------------------
# STEP 7: DEFINE INPUT FEATURES AND TARGET
# ------------------------------------------------------------

# Features used to predict car price
features = [
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner",
    "Year"
]

target = "Selling_Price"


# Check whether all required columns exist
missing_columns = [
    column for column in features + [target]
    if column not in df.columns
]

if missing_columns:
    print("\nERROR!")
    print("The following columns are missing from your dataset:")
    print(missing_columns)
    print("\nAvailable columns are:")
    print(df.columns.tolist())
    exit()


X = df[features]
y = df[target]


# ------------------------------------------------------------
# STEP 8: FEATURE ENGINEERING
# ------------------------------------------------------------

# Create a new feature: Age of the car
current_year = 2026

df["Car_Age"] = current_year - df["Year"]

# Remove Year because Car_Age gives more useful information
features = [
    "Present_Price",
    "Kms_Driven",
    "Car_Age",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner"
]

X = df[features]
y = df[target]


print("\nFeatures used for prediction:")
print(features)


# ------------------------------------------------------------
# STEP 9: IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------

numerical_features = [
    "Present_Price",
    "Kms_Driven",
    "Car_Age",
    "Owner"
]

categorical_features = [
    "Fuel_Type",
    "Seller_Type",
    "Transmission"
]


# ------------------------------------------------------------
# STEP 10: PREPROCESS CATEGORICAL DATA
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ------------------------------------------------------------
# STEP 11: CREATE MACHINE LEARNING PIPELINE
# ------------------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ------------------------------------------------------------
# STEP 12: SPLIT DATA INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))


# ------------------------------------------------------------
# STEP 13: TRAIN MODEL
# ------------------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# ------------------------------------------------------------
# STEP 14: MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# STEP 15: MODEL EVALUATION
# ------------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n" + "=" * 60)
print("                 MODEL EVALUATION")
print("=" * 60)

print(f"\nMean Absolute Error (MAE): {mae:.2f}")

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

print(f"R² Score: {r2:.2f}")

print(
    f"\nModel explains approximately {r2 * 100:.2f}% "
    "of the variation in car prices."
)


# ------------------------------------------------------------
# STEP 16: DISPLAY ACTUAL VS PREDICTED PRICES
# ------------------------------------------------------------

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices:")
print(results.head(10))


# ------------------------------------------------------------
# STEP 17: VISUALIZE ACTUAL VS PREDICTED PRICES
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

# Perfect prediction line
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Car Price")
plt.ylabel("Predicted Car Price")

plt.title("Actual vs Predicted Car Prices")

plt.grid(True)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 18: PREDICT PRICE OF A NEW CAR
# ------------------------------------------------------------

new_car = pd.DataFrame({
    "Present_Price": [5.5],
    "Kms_Driven": [25000],
    "Car_Age": [4],
    "Fuel_Type": ["Petrol"],
    "Seller_Type": ["Dealer"],
    "Transmission": ["Manual"],
    "Owner": [0]
})


predicted_price = model.predict(new_car)

print("\n" + "=" * 60)
print("              NEW CAR PRICE PREDICTION")
print("=" * 60)

print("\nNew car details:")
print(new_car)

print(
    f"\nPredicted Selling Price: "
    f"{predicted_price[0]:.2f} lakh"
)


# ------------------------------------------------------------
# STEP 19: FINISH
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("       CAR PRICE PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 60)