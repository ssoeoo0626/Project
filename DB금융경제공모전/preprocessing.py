import pandas as pd

from sklearn.preprocessing import LabelEncoder

def preprocessing_els(dataset):

    """ 전처리 """
    cols_drop_info = ['종목코드', '기준가1', '녹인가1', '환매일 종가', '평가기준가']
    cols_drop_duplicate = ['평가구분', '상환조건달성'] + ['녹인발생차수'] # unique 값이 1개
    cols_drop_future = ['상환구분', '상환실현차수']

    cols_drop_dt = ['발행일', '상환일', '평가시작일', '평가종료일', '환매결정일', '녹인발생일']
    # 데이터 타입 변경
    for col in cols_drop_dt:
        if col in dataset.columns:
            dataset[col] = pd.to_datetime(dataset[col])

    # 피처로 쓰기 애매함
    cols_cnt = [
        '녹인일수', '녹인일수_전', '영업일수', '상환일수'
    ]

    cols_feature = ['차수', '기초자산개수', '녹인발생차수_차이']

    # 범주형 변수 레이블인코딩
    cols_cat = ['환매일종가위치'] # 범주형 변수
    cols_dummy = ['환매일종가위치_code']
    encoder = LabelEncoder()
    encoder.fit(dataset['환매일종가위치'])
    dataset['환매일종가위치_code'] = encoder.transform(
        dataset['환매일종가위치']
        )
    dataset['환매일종가위치_code'] = dataset['환매일종가위치'].astype('category').cat.codes.astype(float)

    cols_pct100 = [
        '상환조건(%)', '하한 수준(%)', '상환조건감소량(%)_prev', '상환조건감소량(%)_next',
        '환매일 수준(%)', '녹인대비상환수준(%)', '환매대비상환수준(%)', '환매대비상환수준(%)_next'
    ]
    cols_pct = [
        '녹인비율', '녹인비율_전', 'H총증감률', 'H평균증감률', 'H일평균증감률', 'H이전대비증감률', '상환비율'
    ]
    # 비율 단위 변경
    for col in cols_pct:
        col_new = col+"(%)"
        dataset[col_new] = dataset[col]*100
        cols_pct100.append(col_new)
    
    cols_drop = cols_drop_info + cols_drop_duplicate + cols_drop_future \
                + cols_drop_dt + cols_cnt + cols_cat + cols_pct
    return dataset.drop(columns=cols_drop)
    # return dataset[['label'] + cols_feature + cols_dummy + cols_pct100]


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
def scaling(X_train, X_test, cols_cont, method = 'standard'):
    X_train = X_train.copy()
    X_test = X_test.copy()

    if method=='standard':
        scaler = StandardScaler()
    elif method=='robust':
        scaler = RobustScaler()

    scaler.fit(X_train[cols_cont])
    X_train_scaled = scaler.transform(X_train[cols_cont])
    X_train.loc[:, cols_cont] = X_train_scaled # scaling 변수로 대체
    X_test_scaled = scaler.transform(X_test[cols_cont])
    X_test.loc[:, cols_cont] = X_test_scaled
    
    return X_train, X_test