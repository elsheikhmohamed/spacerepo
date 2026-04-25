from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd


train = pd.read_csv('train.csv')

target = train['Transported']
train = train.drop('Transported', axis=1)

object_cols = train.select_dtypes(include=['object']).columns
numeric_cols = train.select_dtypes(include=['number']).columns

# numeric pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# full preprocessing pipeline
full_pipeline = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
])

# transform data
final_train = full_pipeline.fit_transform(train)

# PCA
pca = PCA(n_components=0.95, svd_solver="full")
final_train_pca = pca.fit_transform(final_train)

print(final_train_pca.shape)
