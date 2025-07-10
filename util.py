import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.linear_model import LinearRegression

# Globals
__locations = None
__data_columns = None
__model = None

# Step 1: Train and save artifacts
def train_and_save():
    df = pd.read_csv("model/bengaluru_house_prices.csv")
    df.dropna(inplace=True)

    df['location'] = df['location'].apply(lambda x: x.strip().lower())
    df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]) if pd.notnull(x) else None)

    def convert_sqft(x):
        try:
            if '-' in str(x):
                a, b = x.split('-')
                return (float(a) + float(b)) / 2
            return float(x)
        except:
            return None

    df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
    df.dropna(subset=['total_sqft', 'bath', 'bhk'], inplace=True)

    dummies = pd.get_dummies(df['location'])
    X = pd.concat([df[['total_sqft', 'bath', 'bhk']], dummies], axis=1)
    y = df['price']

    os.makedirs("server/artifacts", exist_ok=True)

    with open("server/artifacts/columns.json", "w") as f:
        json.dump({"data_columns": list(X.columns)}, f)

    model = LinearRegression()
    model.fit(X, y)

    with open("server/artifacts/banglore_home_prices_model.pickle", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and artifacts saved.")


# Step 2: Load model and columns
def load_saved_artifacts():
    print("🔄 Loading saved artifacts...")
    global __data_columns, __locations, __model

    with open("server/artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    if __model is None:
        with open("server/artifacts/banglore_home_prices_model.pickle", "rb") as f:
            __model = pickle.load(f)

    print("✅ Artifacts loaded.")


# Step 3: Predict price
def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

# Step 4: Test run
if __name__ == "__main__":
    train_and_save()
    load_saved_artifacts()
    print(__locations[:5])  # Print few location names
    print(get_estimated_price('1st phase jp nagar', 1000, 3, 3))
    print(get_estimated_price('whitefield', 1200, 2, 2))
    print(get_estimated_price('unknown location', 850, 2, 1))
