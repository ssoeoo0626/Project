# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_model(method = 'lr', random_state=42):
    # lr, dt, svc, rf, xgb, lgbm
    if method=='lr':
        model = LogisticRegression(random_state=random_state)
    elif method=='dt':
        model = DecisionTreeClassifier(random_state=random_state)
    elif method =='svc':
        model = SVC(random_state=random_state, probability=True)
    elif method=='rf':
        model = RandomForestClassifier(random_state=random_state)
    elif method=='xgb':
        model = XGBClassifier(random_state=random_state)
    elif method=='lgbm':
        model = LGBMClassifier(random_state=random_state)
    else:
        return False, 'method는 lr, dt, svc, rf, xgb, lgbm 중 하나여야 함'
    
    return True, model