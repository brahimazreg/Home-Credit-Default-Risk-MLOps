import pandas as pd
from src.config import *
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split


# read data
df=pd.read_csv(DATA_PATH)

# split isolate target from others features
X=df.drop(columns=['SK_ID_CURR','TARGET'],axis=1)
Y=df['TARGET']

# define categoricals and numericals columns 
num_cols = X.select_dtypes(
    include=['int64','float64']
).columns.tolist()

cat_cols = X.select_dtypes(
    include=['object']
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])

def get_train_test_data():

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=Y
    )

    return  X_train, X_test, Y_train, Y_test
 
def get_preprocessor():
    return preprocessor 