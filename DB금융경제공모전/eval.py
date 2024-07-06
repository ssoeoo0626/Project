# 분류 모델 성능평가
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score

def eval(y, y_pred, y_pred_proba=None, print=False):
    # 예측
    # y_pred = model.predict(x)
    # y_pred_proba = model.predict_proba(x)[:, 1]

    cf_matrix = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    try:
        roc_auc = roc_auc_score(y, y_pred_proba)
    except:
        roc_auc=None

    if print:
        print("Confusion Matrix:\n",cf_matrix)
        print("Accuracy : %.3f" % acc)
        print("Precision : %.3f" % prec)
        print("Recall : %.3f" % rec)
        print("F1 : %.3f" % f1)
        print("ROC_AUC :", roc_auc)

    return {
        'actual' : y, 
        'predicted' : y_pred, 
        'predicted_proba' : y_pred_proba, 
        'confusion_matrix' : cf_matrix, 
        'metrics' : {
            'accuracy' : [acc],
            'precision' : [prec], 
            'recall' : [rec],
            'f1' : [f1],
            'roc_auc' : [roc_auc]
        }
    }