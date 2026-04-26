import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- Feature engineering ---
def engineer_features(df):
    df = df.copy()
    # Split Cabin
    df[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = df['Cabin'].str.split('/', expand=True)
    df['Cabin_Num'] = pd.to_numeric(df['Cabin_Num'], errors='coerce')
    
    # Calculate Group/Alone
    group = df['PassengerId'].str.split('_').str[0]
    df['alone'] = (~group.duplicated(keep=False)).astype(int)
    
    # Spending features
    exp_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[exp_features] = df[exp_features].fillna(0)
    df['Total_Spending'] = df[exp_features].sum(axis=1)

    # DROPPING ONLY NON-USEFUL STRINGS (Keep numerical/spending for now)
    return df.drop(columns=['PassengerId', 'Name', 'Cabin'])

# --- Load and Prepare ---
allData = pd.read_csv('train.csv')
allData = engineer_features(allData)

target = allData['Transported'].astype(int)
features = allData.drop('Transported', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# --- Define Column Groups ---
# This is where the error was. We define them as clear lists.
luxury_expenses = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Total_Spending']
categorical_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()
# "Other" numbers are numeric columns NOT in the luxury list
numeric_cols = [col for col in X_train.select_dtypes(include=['number']).columns if col not in luxury_expenses]

# --- Preprocessing ---
exp_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), 
    ("log", FunctionTransformer(np.log1p)),       
    ("scaler", StandardScaler())
])

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("exp", exp_pipeline, luxury_expenses),
    ("num", num_pipeline, numeric_cols),
    ("cat", cat_pipeline, categorical_cols),
])

# --- Full model pipeline ---
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(
        random_state=42,
        max_iter=500,             
        learning_rate=0.05,      
        max_depth=5,             
        l2_regularization=1.5,    
        early_stopping=True
    )),
])

# --- Evaluation ---
x_pred = cross_val_predict(model, X_train, y_train, cv=5)
print("Confusion Matrix:")
print(confusion_matrix(y_train, x_pred))

model.fit(X_train, y_train)
print(f"\nTrain accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test accuracy:  {model.score(X_test, y_test):.4f}")