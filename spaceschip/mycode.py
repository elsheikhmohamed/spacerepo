from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd


# --- Feature engineering ---
def engineer_features(df):
    df = df.copy()
    df[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = df['Cabin'].str.split('/', expand=True)
    df['Cabin_Num'] = pd.to_numeric(df['Cabin_Num'], errors='coerce')
    group = df['PassengerId'].str.split('_').str[0]
    df['alone'] = (~group.duplicated(keep=False)).astype(int)
    return df.drop(columns=['PassengerId', 'Name', 'Cabin'])


allData = pd.read_csv('train.csv')
allData = engineer_features(allData)


target = allData['Transported'].astype(int)
features = allData.drop('Transported', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()


# --- Preprocessing ---
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])




# --- Full model pipeline ---
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(
        random_state=42,
        max_iter=500,             # Allow more iterations...
        learning_rate=0.05,       # ...but learn slower to avoid overfitting
        max_depth=5,              # Strictly limit tree complexity
        min_samples_leaf=20,      # Require at least 20 passengers per final leaf
        l2_regularization=1.5,    # Penalize overly complex rules
        early_stopping=True,      # Stop automatically if it starts overfitting
        validation_fraction=0.1   # Use 10% of training data as the early-stopping monitor
    )),
])

x= cross_val_predict(model, X_train, y_train, cv=5)
print(confusion_matrix(y_train, x))
model.fit(X_train, y_train)

print(f"Train accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test accuracy:  {model.score(X_test, y_test):.4f}")