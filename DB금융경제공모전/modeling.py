import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score
import seaborn as sns
from sklearn.model_selection import train_test_split

rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

file = 'dataset.csv'

dataset = pd.read_csv(file)
print(dataset.shape)
# dataset.head().transpose()


""" 전처리 """
cols_drop_info = ['종목코드', '기준가1', '녹인가1', '환매일 종가', '평가기준가']
cols_drop_duplicate = ['평가구분', '상환조건달성'] + ['녹인발생차수'] # unique 값이 1개
cols_drop_future = ['상환구분', '상환실현차수']

cols_drop_dt = ['발행일', '상환일', '평가시작일', '평가종료일', '환매결정일', '녹인발생일']
# 데이터 타입 변경
for col in cols_drop_dt:
    dataset[col] = pd.to_datetime(dataset[col])


# 피처로 쓰기 애매함
cols_cnt = [
    '녹인일수', '녹인일수_전', '영업일수', '상환일수'
]

cols_feature = ['차수', '기초자산개수', '녹인발생차수_차이']
cols_cat = ['환매일종가위치'] # 범주형 변수
cols_dummy = ['환매일종가위치_code']
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

#
cols_drop = cols_drop_info + cols_drop_duplicate + cols_drop_future + cols_drop_dt + cols_cat
dataset['환매결정일'] = pd.to_datetime(dataset['환매결정일'])
dataset.set_index('환매결정일', inplace= True)

long = pd.read_csv('장기피처_합치기_is.csv')
long.rename(columns= {'Unnamed: 0':'환매결정일'}, inplace= True)
long['환매결정일'] = pd.to_datetime(long['환매결정일'])
short = pd.read_csv('final_total.csv')
short.rename(columns= {'Unnamed: 0':'환매결정일'}, inplace= True)
short['환매결정일'] = pd.to_datetime(short['환매결정일'])
short.set_index('환매결정일', inplace= True)

dfs = [long, short]
dataset_merge = dataset.copy()
for df in dfs:
    dataset_merge = pd.merge(
        dataset_merge, df,
        left_index=True, right_index=True,
        how='inner'
    )
dataset = dataset_merge

cat_features = [
    'PMI_dummy','CLI_BRA_dummy','CLI_CAN_dummy','CLI_CHN_dummy',
    'CLI_DEU_dummy','CLI_ESP_dummy','CLI_FRA_dummy','CLI_GBR_dummy',
    'CLI_G-20_dummy','CLI_G-7_dummy','CLI_IDN_dummy','CLI_ITA_dummy',
    'CLI_JPN_dummy','CLI_KOR_dummy','CLI_MEX_dummy','CLI_TUR_dummy',
    'CLI_USA_dummy','CLI_ZAF_dummy']

dataset_values = dataset.reset_index().drop(columns=cols_drop+cols_pct)
dataset_values.corr()['label'].sort_values()*100
# col_X = cols_feature + cols_pct100
col_X = dataset_values.columns.difference(['label'])
col_y = 'label'

# df = dataset[col_X + [col_y]]
df = dataset_values

X = dataset_values[col_X]
y = dataset_values[col_y]

### 3번 선택된 피쳐 모듈 추가

X = dataset_values[[ 'CLI_BRA_dummy',
 'CLI_KOR',
 'CLI_TUR_pct',
 'CLI_ZAF_pct',
 'H이전대비증감률(%)',
 'H일평균증감률(%)',
 'Oman',
 'PMI_dummy',
 'WTI(%)',
 'cpi_ind(%)_shift',
 'cpi_usa(%)_shift',
 'ind_cci(%)_shift',
 'kor_cci_100기준_shift',
 'msci_China(USD)(%)',
'국채 수익률',
 '녹인대비상환수준(%)',
 '녹인일수_전',
 '변동',
 '상환조건(%)',
 '상환조건감소량(%)_next',
 '영업일수',
 '홍콩 M2 (2).1(%)_shift',
 '홍콩CPI',
 '홍콩CPI_shift',
 '홍콩CPI_shift_diff()',
 '홍콩수출액',
 '홍콩수출액_shift',
 '홍콩수출액_shift_diff',
 '홍콩외환보유액(%)',
 '환매대비상환수준(%)',
 '환매대비상환수준(%)_next']]

from sklearn.preprocessing import StandardScaler

# Assuming cols_feature_long is your DataFrame
# Select only the numerical columns for scaling
columns_to_scale = [[# 'CLI_BRA_dummy',
 'CLI_KOR',
 'CLI_TUR_pct',
 'CLI_ZAF_pct',
 'H이전대비증감률(%)',
 'H일평균증감률(%)',
 'Oman',
 #'PMI_dummy',
 'WTI(%)',
 'cpi_ind(%)_shift',
 'cpi_usa(%)_shift',
 'ind_cci(%)_shift',
 'kor_cci_100기준_shift',
 'msci_China(USD)(%)',
 '국채 수익률',
 '녹인대비상환수준(%)',
 '녹인일수_전',
 '변동',
 '상환조건(%)',
 '상환조건감소량(%)_next',
 '영업일수',
 '홍콩 M2 (2).1(%)_shift',
 '홍콩CPI',
 '홍콩CPI_shift',
 '홍콩CPI_shift_diff()',
 '홍콩수출액',
 '홍콩수출액_shift',
 '홍콩수출액_shift_diff',
 '홍콩외환보유액(%)',
 '환매대비상환수준(%)',
 '환매대비상환수준(%)_next']]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify= y)

# Create an instance of StandardScaler
scaler = StandardScaler()

# Fit the scaler to the selected columns and transform them
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def preprocess_data(X_train, X_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    return accuracy, precision, recall, f1

def model_basic(X_train, y_train, X_val, y_val):
    models = [
        LogisticRegression(),
        DecisionTreeClassifier(),
        SVC(),
        RandomForestClassifier(),
        XGBClassifier(),
        LGBMClassifier()
    ]

    results_dict = {'model':[], 'accuracy':[], 'precision':[], 'recall':[], 'f1_score':[]}

    for clf in models:
        X_train_scaled, X_val_scaled = preprocess_data(X_train, X_val)
        accuracy, precision, recall, f1 = train_and_evaluate_model(clf, X_train_scaled, y_train, X_val_scaled, y_val)

        results_dict['model'].append(clf)
        results_dict['accuracy'].append(accuracy)
        results_dict['precision'].append(precision)
        results_dict['recall'].append(recall)
        results_dict['f1_score'].append(f1)

    results_df = pd.DataFrame(data=results_dict)
    return results_df