from sklearn.metrics import accuracy_score ,recall_score, precision_score , f1_score , confusion_matrix
import pandas as pd
def compute_metrics(y_true,y_pred):
    if len(y_true) != len(y_pred) : 
        raise ValueError("lenghths does not match")
    if len(y_true) == 0:
        raise ValueError("Evaluation inputs cannot be empty")

    if not set(y_true).issubset({0, 1}) or not set(y_pred).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1")
    
    accuracy=accuracy_score(y_true,y_pred)
    precision=precision_score(y_true,y_pred,zero_division=0)
    recall=recall_score(y_true,y_pred,zero_division=0)
    f1=f1_score(y_true,y_pred,zero_division=0)
    confusion=confusion_matrix(y_true,y_pred,labels=[0, 1])
    return {
        "accuracy" : accuracy,
        "precision" : precision,
        "recall" : recall,
        "f1" : f1,
        "confusion_matrix" : confusion
    }

def analyze_errors(dataset,predictions , confidences):
    if not (len(dataset) == len(predictions) == len(confidences)):
        raise ValueError("Input lengths do not match")

    dataset_modified=dataset.add_column(column=predictions , name="predicted")
    dataset_modified=dataset_modified.add_column(column=confidences , name="confidence")
    dataset_modified=dataset_modified.rename_column("label" , "actual")
    df=pd.DataFrame(dataset_modified)
    df["correct"]=(df["actual"] == df["predicted"])

    errors=df[df["correct"]==False]
    false_positive=df[
       ( df["actual"] == 0) & (df["predicted"] == 1)
    ]

    false_negative=df[
           ( df["actual"] == 1) & (df["predicted"] == 0)
        ]
    errors=errors.sort_values("confidence" , ascending=False)
    worst_errors=errors.head(10)
    return {
        "all_predictions": df,
        "errors": errors,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "high_confidence_errors": worst_errors
    }
