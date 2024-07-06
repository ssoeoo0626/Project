import pandas as pd

from . import models

# logit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.api as sm
import numpy as np
def logit(X_train, y_train):
    lr_clf = LogisticRegression()
    feature = X_train
    target = y_train

    logit = SelectFromModel(LogisticRegression())
    logit.fit(feature, target)
    logit_support = logit.get_support()
    lr_feature = feature.loc[:,logit_support].columns.tolist()
    return lr_feature

def vif(data):
    # VIF 출력을 위한 데이터 프레임 형성
    vif = pd.DataFrame()

    # VIF 값과 각 Feature 이름에 대해 설정
    vif["VIF Factor"] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]
    vif["features"] = data.columns 
                    
    # VIF 값이 높은 순으로 정렬
    vif = vif.sort_values(by="VIF Factor", ascending=False)
    vif = vif.reset_index().drop(columns='index')
    
    return vif

def filtering_vif(data, thr=10):
    vifs = vif(data)
    # vif=1 : corr=1인 거임
    print("필터링 전 피처 수:", vifs.shape[0])
    while max(vifs['VIF Factor']) >= thr:
        max_feature = vifs.iloc[0, -1]
        print(f"max(VIF): {vifs['VIF Factor'].max()}, 제거할 변수:{max_feature}")
        data = data.drop(columns = [max_feature])
        vifs = vif(data)
    print("필터링 후 피처 수:", vifs.shape[0])
    return data, vifs

    
# vif(train[lr_feature])

# 릿지

# t-test



# Forward feature selection 수행
def forward(X_train, X_val, y_train, y_val, method='lr'):
    selected_features_forward = []
    best_score = 0
    
    #무한 반복 루프 전체 특성의 개수보다 선택된 특성의 개수가 작을때 까지 반복
    while len(selected_features_forward) < X_train.shape[1]: 
        best_feature = None     # 가장좋은 특성 이름
        best_model = None       # 가장좋은 모델 저장
        best_score_local = 0    # 가장 높은 정확도

        for feature in X_train.columns:
            if feature not in selected_features_forward:
                features = selected_features_forward + [feature]
                X_train_selected = X_train[features]
                # return X_train_selected
                X_val_selected = X_val[features]

                # model = LogisticRegression()
                stat, model = models.get_model(method = method)
                if not stat:
                    print("method 설정이 잘못됨")
                    return
                model.fit(X_train_selected, y_train)
                score = model.score(X_val_selected, y_val)

                if score > best_score_local:
                    best_score_local = score
                    best_feature = feature
                    best_model = model

        if best_score_local > best_score:
            selected_features_forward.append(best_feature)
            best_score = best_score_local
            print(f"Selected feature: {best_feature}, Accuracy: {best_score:.4f}")

        else:
            break

    print("\nForward selected features:")
    Forward = selected_features_forward
    print(Forward)
    return Forward

# Backward Elimination
def backward(X_train, X_val, y_train, y_val, method='lr'):

    # Backward feature selection 수행
    selected_features_backward = X_train.columns.tolist()
    best_score = 0

    while len(selected_features_backward) > 0:
        worst_feature = None
        best_model = None
        best_score_local = 0

        for feature in selected_features_backward:
            features = selected_features_backward.copy()
            features.remove(feature)

            X_train_selected = X_train[features]
            X_val_selected = X_val[features]

            # model = LogisticRegression()
            stat, model = models.get_model(method = method)
            if not stat:
                print("method 설정이 잘못됨")
                return
            model.fit(X_train_selected, y_train)
            score = model.score(X_val_selected, y_val)

            if score > best_score_local:
                best_score_local = score
                worst_feature = feature
                best_model = model

        if best_score_local > best_score:
            selected_features_backward.remove(worst_feature)
            best_score = best_score_local
            print(f"Removed feature: {worst_feature}, Accuracy: {best_score:.4f}")

        else:
            break

    print("\nBackward selected features:")
    Backward = selected_features_backward
    print(Backward)
    return Backward

# Stepwise feature selection 수행
def stepwise(X_train, X_val, y_train, y_val, method='lr'):
    selected_features_stepwise = []
    best_score = 0

    # Forward step
    while len(selected_features_stepwise) < X_train.shape[1]:
        best_feature = None
        best_model = None
        best_score_local = 0

        for feature in X_train.columns:
            if feature not in selected_features_stepwise:
                features = selected_features_stepwise + [feature]
                X_train_selected = X_train[features]
                X_val_selected = X_val[features]

                # model = LogisticRegression()
                stat, model = models.get_model(method = method)
                if not stat:
                    print("method 설정이 잘못됨")
                    return

                model.fit(X_train_selected, y_train)
                score = model.score(X_val_selected, y_val)

                if score > best_score_local:
                    best_score_local = score
                    best_feature = feature
                    best_model = model

        if best_score_local > best_score:
            selected_features_stepwise.append(best_feature)
            best_score = best_score_local
            print(f"Selected feature: {best_feature}, Accuracy: {best_score:.4f}")

        else:
            break

    # Backward step
    while len(selected_features_stepwise) > 0:
        worst_feature = None
        best_model = None
        best_score_local = 0

        for feature in selected_features_stepwise:
            features = selected_features_stepwise.copy()
            features.remove(feature)

            X_train_selected = X_train[features]
            X_val_selected = X_val[features]

            # model = LogisticRegression()
            stat, model = models.get_model(method = method)
            if not stat:
                print("method 설정이 잘못됨")
                return

            model.fit(X_train_selected, y_train)
            score = model.score(X_val_selected, y_val)

            if score > best_score_local:
                best_score_local = score
                worst_feature = feature
                best_model = model

        if best_score_local > best_score:
            selected_features_stepwise.remove(worst_feature)
            best_score = best_score_local
            print(f"Removed feature: {worst_feature}, Accuracy: {best_score:.4f}")

        else:
            break

    print("\nStepwise selected features:")
    Stepwise = selected_features_stepwise

    print(Stepwise)
    return Stepwise

