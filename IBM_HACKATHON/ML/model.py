import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle

train = pd.read_csv(r"C:\Users\Ananth\Downloads\train.csv")

train2 = train.drop(
    [
        "Date_of_creation",
        "Contact_no",
        "POC_name",
        "Lead_POC_email",
        "Internal_POC",
        "Deal_title",
        "Lead_name",
    ],
    axis=1,
)


train2["Designation"] = train2["Designation"].replace(
    {
        "Chairman/CEO/President": "Chairman/CEO/President",
        "CEO/Chairman/President": "Chairman/CEO/President",
        "Chief Executive Officer": "CEO",
        "Vice President / GM (04-present) : VP Sales and Marketing (01-04)": "Vice President/GM",
    }
)


train2["Last_lead_update"].replace("?", "No track", inplace=True)
train2["Last_lead_update"].replace(np.nan, "No track", inplace=True)


train2 = train2.dropna(
    how="any", subset=["Industry", "Resource", "Deal_value", "Location"], axis=0
)


train2["Geography"] = train2.apply(
    lambda row: "USA"
    if pd.isna(row["Geography"]) and "," in row["Location"]
    else ("India" if pd.isna(row["Geography"]) else row["Geography"]),
    axis=1,
)
train2["Level_of_meeting"] = train2.apply(
    lambda row: 1
    if "1" in row["Level_of_meeting"]
    else (2 if "2" in row["Level_of_meeting"] else 3),
    axis=1,
)


train2["Deal_value"] = train2["Deal_value"].str.replace("$", "")
train2["Weighted_amount"] = train2["Weighted_amount"].str.replace("$", "")
train2["Weighted_amount"] = train2["Weighted_amount"].astype(float)
train2["Deal_value"] = train2["Deal_value"].astype(float)
train2["Weighted_amount"].fillna(train2["Weighted_amount"].mean(), inplace=True)


train2.drop("Location", axis=1, inplace=True)


train2 = pd.get_dummies(
    train2,
    columns=[
        "Industry",
        "Pitch",
        "Lead_revenue",
        "Fund_category",
        "Geography",
        "Designation",
        "Hiring_candidate_role",
        "Lead_source",
        "Last_lead_update",
        "Resource",
    ],
)

# Select only the numerical columns
numerical_columns = ["Deal_value", "Weighted_amount"]


# Function to remove outliers based on a threshold
def remove_outliers(df, columns, threshold=2):
    df_no_outliers = df.copy()
    for column in columns:
        mean = df[column].mean()
        std = df[column].std()
        lower_bound = mean - (threshold * std)
        upper_bound = mean + (threshold * std)
        df_no_outliers = df_no_outliers[
            (df_no_outliers[column] >= lower_bound)
            & (df_no_outliers[column] <= upper_bound)
        ]
    return df_no_outliers


# Remove outliers from numerical columns
train2 = remove_outliers(train2, numerical_columns)
# print(train2.describe())
# print(train2[["Success_probability"]].median())


X = train2.drop("Success_probability", axis=1)
y = train2["Success_probability"]

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create a Random Forest Regressor
n_estimators = 100  # Number of trees in the forest
max_depth = None  # Maximum depth of the individual trees
rf_regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

weights = np.where((y >= 60) & (y <= 80), 0.2, 0.8)
# Train the Random Forest Regressor
rf_regressor.fit(X, y, sample_weight=weights)


pickle.dump(rf_regressor, open("ml_model.sav", "wb"))
