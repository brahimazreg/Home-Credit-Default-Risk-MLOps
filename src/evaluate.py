from sklearn.metrics import classification_report,roc_auc_score,recall_score,f1_score,accuracy_score
from joblib import load
from src.data_preprocessing import get_train_test_data
from src.config import *


X_train, X_test, Y_train, Y_test = get_train_test_data()

model = load(default_model)

def evaluate():
    Y_pred=model.predict(X_test)

    accuracy = accuracy_score(Y_test,Y_pred)
    recallscore = recall_score(Y_test,Y_pred)
    proba = model.predict_proba(X_test)[:,1]
    roc_auc = roc_auc_score(Y_test, proba)
    f1score=f1_score(Y_test,Y_pred)
    report = classification_report(Y_test , Y_pred)
    return  accuracy,recallscore , roc_auc  , report , f1score

if __name__ == "__main__":
    accuracy,recallscore , roc_auc  , report , f1score = evaluate()

    print(f"accuracy: {accuracy :.4f}")
    print(f"recall: {recallscore :.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")    
    print(f"f1score: {f1score:.4f}")
    print(report)
