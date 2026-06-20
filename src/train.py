import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline as skpipeline
from imblearn.over_sampling import SMOTE
from joblib import dump
from src.data_preprocessing import get_train_test_data, get_preprocessor
from src.config import MODEL_PATH, RANDOM_STATE,default_model
import xgboost
from xgboost import XGBClassifier

# save model
#-----------
def save_model(model):
    model_name = MODEL_PATH / f"{model.named_steps['model'].__class__.__name__}.joblib"     
    model_name.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_name)
    print(f"Model saved to {model_name}")  

# ---------------------------
# LOAD DATA ONCE
# ---------------------------
X_train, X_test, Y_train, Y_test = get_train_test_data()
preprocessor = get_preprocessor()

# thishelp to chose parameters inxgboost
neg = (Y_train == 0).sum()
pos = (Y_train == 1).sum()

print(f"Negative samples: {neg}")
print(f"Positive samples: {pos}")
print(f"Imbalance ratio: {neg / pos:.2f}")

# ---------------------------
# MLflow setup
# ---------------------------
mlflow.sklearn.autolog(disable=True)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("home_credit_prediction")


def train_data(model_name, model):

    with mlflow.start_run(run_name=model.__class__.__name__):

        pipeline = imbpipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model)
        ])

        pipeline.fit(X_train, Y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]


        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        dump(pipeline, MODEL_PATH / f"{model.__class__.__name__}.joblib")
        print(f"Saved locally: {model.__class__.__name__}.joblib")

        # metrics
        mlflow.log_metrics({
            "accuracy": accuracy_score(Y_test, y_pred),
            "roc_auc": roc_auc_score(Y_test, y_proba),
            "f1_score": f1_score(Y_test, y_pred),
            "recall": recall_score(Y_test, y_pred),
            "precision": precision_score(Y_test, y_pred)
        })

        model_registry = {
            "lr": "home_credit_lr",
            "rf": "home_credit_rf",
            "xgb": "home_credit_xgb"
        }

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=model_registry[model_name],
            serialization_format="cloudpickle"
        )

        return pipeline


# ---------------------------
# MODEL
# ---------------------------
models = {
    "lr": LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000
    ),
    "rf": RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=300,
        max_depth =6,
    ),
    "xgb": XGBClassifier(
    random_state=RANDOM_STATE,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=1,
    eval_metric="logloss"
    )
}


if __name__ == "__main__":
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        train_data(model_name ,model)
    #train_data("xgb", models["xgb"])