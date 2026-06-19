import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline as skpipeline
from imblearn.over_sampling import SMOTE
from joblib import dump
from src.data_preprocessing import get_train_test_data, get_preprocessor
from src.config import MODEL_PATH, RANDOM_STATE

# save model
#-----------
def save_model(model):
    #model_name = MODEL_PATH / f"{model.named_steps['model'].__class__.__name__}.joblib"
   
    model_name = MODEL_PATH / "logistic_regression.joblib"
    model_name.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_name)

    print(f"Model saved to {model_name}")

# ---------------------------
# LOAD DATA ONCE
# ---------------------------
X_train, X_test, Y_train, Y_test = get_train_test_data()
preprocessor = get_preprocessor()


# ---------------------------
# MLflow setup
# ---------------------------
mlflow.sklearn.autolog(disable=True)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("home_credit_prediction")


def train_data(model):

    with mlflow.start_run():

        # ---------------------------
        # BUILD PIPELINE
        # ---------------------------
         
        training_pipeline = imbpipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model)
        ])

        training_pipeline.fit(X_train, Y_train)
        trained_model = training_pipeline.named_steps["model"] 
    
        fitted_preprocessor = training_pipeline.named_steps["preprocessor"]
        inference_pipeline = skpipeline([
            ("preprocessor", fitted_preprocessor),
            ("model", trained_model)
        ])    
        save_model(inference_pipeline)

        # ---------------------------
        # PREDICTIONS
        # ---------------------------
        y_pred = training_pipeline.predict(X_test)
        y_proba = training_pipeline.predict_proba(X_test)[:, 1]

        # ---------------------------
        # METRICS
        # ---------------------------
        accuracy = accuracy_score(Y_test, y_pred)
        roc_auc = roc_auc_score(Y_test, y_proba)
        f1 = f1_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        
        # ---------------------------
        # PARAMS
        # ---------------------------
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", model.max_iter)
        mlflow.log_param("random_state", RANDOM_STATE)

        # ---------------------------
        # SAVE MODEL (IMPORTANT)
        # ---------------------------
        mlflow.sklearn.log_model(
            sk_model=inference_pipeline,
            artifact_path="model",
            registered_model_name="home_credit_model",
            serialization_format="cloudpickle"
        )

        print("Training completed & logged to MLflow")

        return training_pipeline


# ---------------------------
# MODEL
# ---------------------------
models = {
    "lr": LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000
    )
}


if __name__ == "__main__":
    train_data(models["lr"])