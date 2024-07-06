import warnings
warnings.filterwarnings('ignore')

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import rc
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

import pandas as pd
long = pd.read_csv('장기피처_합치기_is.csv')
long.rename(columns= {'Unnamed: 0':'환매결정일'}, inplace= True)
long['환매결정일'] = pd.to_datetime(long['환매결정일'])
long.set_index('환매결정일', inplace= True)
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

cat_features = ['PMI_dummy', 'CLI_BRA_dummy', 'CLI_CAN_dummy',
    'CLI_CHN_dummy','CLI_DEU_dummy','CLI_ESP_dummy','CLI_FRA_dummy',
    'CLI_GBR_dummy','CLI_G-20_dummy','CLI_G-7_dummy',
    'CLI_IDN_dummy','CLI_ITA_dummy','CLI_JPN_dummy','CLI_KOR_dummy',
    'CLI_MEX_dummy','CLI_TUR_dummy','CLI_USA_dummy','CLI_ZAF_dummy']

dataset_values = dataset.reset_index().drop(columns=cols_drop+cols_pct)
dataset_values.corr()['label'].sort_values()*100

# col_X = cols_feature + cols_pct100
col_X = dataset_values.columns.difference(['label'])
col_y = 'label'

# df = dataset[col_X + [col_y]]
df = dataset_values

X = dataset_values[col_X]
y = dataset_values[col_y]

df.shape, X.shape, y.shape

columns_to_scale = ['Brent', 'Brent(%)', 'CLI_BRA', 'CLI_BRA_dummy', 'CLI_BRA_pct', 'CLI_CAN', 'CLI_CAN_dummy', 'CLI_CAN_pct', 'CLI_CHN', 'CLI_CHN_dummy', 'CLI_CHN_pct', 'CLI_DEU', 'CLI_DEU_dummy', 'CLI_DEU_pct', 'CLI_ESP', 'CLI_ESP_dummy', 'CLI_ESP_pct', 'CLI_FRA', 'CLI_FRA_dummy', 'CLI_FRA_pct', 'CLI_G-20', 'CLI_G-20_dummy', 'CLI_G-20_pct', 'CLI_G-7', 'CLI_G-7_dummy', 'CLI_G-7_pct', 'CLI_GBR', 'CLI_GBR_dummy', 'CLI_GBR_pct', 'CLI_IDN', 'CLI_IDN_dummy', 'CLI_IDN_pct', 'CLI_ITA', 'CLI_ITA_dummy', 'CLI_ITA_pct', 'CLI_JPN', 'CLI_JPN_dummy', 'CLI_JPN_pct', 'CLI_KOR', 'CLI_KOR_dummy', 'CLI_KOR_pct', 'CLI_MEX', 'CLI_MEX_dummy', 'CLI_MEX_pct', 'CLI_TUR', 'CLI_TUR_dummy', 'CLI_TUR_pct', 'CLI_USA', 'CLI_USA_dummy', 'CLI_USA_pct', 'CLI_ZAF', 'CLI_ZAF_dummy', 'CLI_ZAF_pct', 'Dubai', 'Dubai(%)', 'HKPPI_shift', 'H이전대비증감률(%)', 'H일평균증감률(%)', 'H총증감률(%)', 'H평균증감률(%)', 'IIP(%)', 'MSCI_China(USD)', 'Oman', 'Oman(%)', 'PMI_dummy', 'PMI_변동(%)', 'PMI_차이', 'RGDP(%)', 'WTI', 'WTI(%)', 'chn_cci(%)_shift', 'chn_cci_100기준_shift', 'chn_cci_shift', 'cpi_chn(%)_shift', 'cpi_chn_shift', 'cpi_ind(%)_shift', 'cpi_ind_shift', 'cpi_jap(%)_shift', 'cpi_jap_shift', 'cpi_kor(%)_shift', 'cpi_kor_shift', 'cpi_usa(%)_shift', 'cpi_usa_shift', 'ind_cci(%)_shift', 'ind_cci_100기준_shift', 'ind_cci_shift', 'kor_cci(%)_shift', 'kor_cci_100기준_shift', 'kor_cci_shift', 'msci_China(USD)(%)', '국채 수익률', '국채 수익률증감', '기초자산개수', '녹인대비상환수준(%)', '녹인발생차수_차이', '녹인비율(%)', '녹인비율_전(%)', '녹인일수', '녹인일수_전', '대체투자_shift', '대체투자부채_shift', '대체투자자산_shift', '대출금리_chn', '대출금리_hk', '대출금리_ind', '대출금리_kor', '대출금리_sing', '대출금리_tai', '대출금리_us', '변동', '상환비율(%)', '상환일수', '상환조건(%)', '상환조건감소량(%)_next', '상환조건감소량(%)_prev', '시장금리_hk', '시장금리_ind', '시장금리_kor', '시장금리_sing', '시장금리_tai', '시장금리_us', '영업일수', '예금금리_chn', '예금금리_hk', '예금금리_kor', '예금금리_sing', '예금금리_tai', '일일증권평균거래량_Main Board_Other equity stocks(%)_shift', '일일증권평균거래량_Main Board_Other equity stocks_shift', '일일증권평균거래량_Main Board_Other listed securities_shift', '일일증권평균거래량_Main Board_Sub-total_shift', '차수', '초단기 국채율_hk', '초단기 국채율_tai', '초단기 국채율_us', '하한 수준(%)', '홍콩 M1 (seasonally adjusted)_shift', '홍콩 M1 외국통화.1_shift', '홍콩 M1 자국통화_shift', '홍콩 M1 종합_shift', '홍콩 M2 (2).1(%)_shift', '홍콩 M2 (adjusted for foreign currency swap deposits).1_shift', '홍콩 M2 (adjusted for foreign currency swap deposits).2_shift', '홍콩 M2 (adjusted for foreign currency swap deposits)_shift', '홍콩 M2 외국통화.1_shift', '홍콩 M2 자국통화_shift', '홍콩 M2 종합_shift', '홍콩 M3 (adjusted for foreign currency swap deposits).1_shift', '홍콩 M3 (adjusted for foreign currency swap deposits).2_shift', '홍콩 M3 (adjusted for foreign currency swap deposits)_shift', '홍콩 M3 외국통화.1_shift', '홍콩 M3 자국통화_shift', '홍콩 M3 종합_shift', '홍콩CPI', '홍콩CPI_shift', '홍콩CPI_shift_diff()', '홍콩거래소시가총액_Main BoardH Shares(%)_shift', '홍콩거래소시가총액_Main BoardH Shares_shift', '홍콩거래소시가총액_Main Board_Others_shift', '홍콩거래소시가총액_Main Board_Sub-total_shift', '홍콩금융계좌_shift', '홍콩금융예비자산_shift', '홍콩기계장비PPI_shift', '홍콩기타제조업PPI(1)_shift', '홍콩수입액', '홍콩수입액_shift', '홍콩수입액_shift_diff', '홍콩수출액', '홍콩수출액_shift', '홍콩수출액_shift_diff', '홍콩예비자산_shift', '홍콩외환보유액', '홍콩외환보유액(%)', '홍콩외환보유액_shift', '홍콩음식료PPI_shift', '홍콩의류PPI_shift', '홍콩인쇄미디어PPI_shift', '홍콩직접투자(%)_shift', '홍콩직접투자(%)_shift.1', '홍콩직접투자_shift', '홍콩직접투자_shift.1', '홍콩직접투자부채_shift', '홍콩직접투자자산_shift', '홍콩파생상품(%)_shift', '홍콩파생상품_shift', '홍콩파생상품부채_shift', '홍콩파생상품자산_shift', '홍콩포트폴리오투자부채_shift', '홍콩포트폴리오투자자산_shift', '환매대비상환수준(%)', '환매대비상환수준(%)_next', '환매일 수준(%)', '환매일종가위치_code']

#### wrapper

# Forward Selection

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 학습 데이터와 검증 데이터 분리
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Forward feature selection 수행
selected_features = []
best_score = 0

while len(selected_features) < X.shape[1]: #무한 반복 루프 전체 특성의 개수보다 선택된 특성의 개수가 작을때 까지 반복
    best_feature = None #가장좋은 특성 이름
    best_model = None #가장좋은 모델 저장
    best_score_local = 0 #가장 높은 정확도

    for feature in X_train.columns:
        if feature not in selected_features:
            features = selected_features + [feature]
            X_train_selected = X_train[features]
            X_val_selected = X_val[features]

            model = LogisticRegression()
            model.fit(X_train_selected, y_train)
            score = model.score(X_val_selected, y_val)

            if score > best_score_local:
                best_score_local = score
                best_feature = feature
                best_model = model

    if best_score_local > best_score:
        selected_features.append(best_feature)
        best_score = best_score_local
        print(f"Selected feature: {best_feature}, Accuracy: {best_score:.4f}")

    else:
        break

Forward = selected_features

# Backward Elimination
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 학습 데이터와 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42,stratify= y)

# Backward feature selection 수행
selected_features = X_train.columns.tolist()
best_score = 0

while len(selected_features) > 0:
    worst_feature = None
    best_model = None
    best_score_local = 0

    for feature in selected_features:
        features = selected_features.copy()
        features.remove(feature)

        X_train_selected = X_train[features]
        X_val_selected = X_val[features]

        model = LogisticRegression()
        model.fit(X_train_selected, y_train)
        score = model.score(X_val_selected, y_val)

        if score > best_score_local:
            best_score_local = score
            worst_feature = feature
            best_model = model

    if best_score_local > best_score:
        selected_features.remove(worst_feature)
        best_score = best_score_local
        print(f"Removed feature: {worst_feature}, Accuracy: {best_score:.4f}")

    else:
        break

Backward = selected_features

# Stepwise Selection
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 학습 데이터와 검증 데이터 분리
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify= y)

# Stepwise feature selection 수행
selected_features = []
best_score = 0

# Forward step
while len(selected_features) < X.shape[1]:
    best_feature = None
    best_model = None
    best_score_local = 0

    for feature in X_train.columns:
        if feature not in selected_features:
            features = selected_features + [feature]
            X_train_selected = X_train[features]
            X_val_selected = X_val[features]

            model = LogisticRegression()
            model.fit(X_train_selected, y_train)
            score = model.score(X_val_selected, y_val)

            if score > best_score_local:
                best_score_local = score
                best_feature = feature
                best_model = model

    if best_score_local > best_score:
        selected_features.append(best_feature)
        best_score = best_score_local
        print(f"Selected feature: {best_feature}, Accuracy: {best_score:.4f}")

    else:
        break

# Backward step
while len(selected_features) > 0:
    worst_feature = None
    best_model = None
    best_score_local = 0

    for feature in selected_features:
        features = selected_features.copy()
        features.remove(feature)

        X_train_selected = X_train[features]
        X_val_selected = X_val[features]

        model = LogisticRegression()
        model.fit(X_train_selected, y_train)
        score = model.score(X_val_selected, y_val)

        if score > best_score_local:
            best_score_local = score
            worst_feature = feature
            best_model = model

    if best_score_local > best_score:
        selected_features.remove(worst_feature)
        best_score = best_score_local

    else:
        break

Stepwise = selected_features

# H가 1인 경우 이분산성 / H가 0인 경우 등분산
from scipy.stats import bartlett
def bartlett_test(col, p_value = 0.05, H = 1):
    list= []
    for i in col:
        T, p_val =bartlett(df[df['label']==1][i], df[df['label']==0][i]) 
        list.append([i, p_val])

    list = pd.DataFrame(list, columns = ['변수', 'p_value'])
    if H == 1:
        a = list[(list['p_value'] < p_value)][['변수', 'p_value']].sort_values('p_value')
        return a
    else:
        a = list[(list['p_value'] >= p_value)][['변수', 'p_value']].sort_values('p_value')
        return a
    
# 이분산성 변수
x_hetero = bartlett_test(df.columns, H = 1)
# 등분산성 변수
x_homo = bartlett_test(df.columns, H = 0)

# t-test

import scipy.stats as stats
def t_test(col, col_h0, col_h1, p_value = 0.05):
    list= []
    for i in col:
        if (col_h0['변수']==i).any():
            t_stat, p_val = stats.ttest_ind(df[df['label']==1][i], df[df['label']==0][i], equal_var=True) # 등분산성 : wald t-test
            list.append([i, p_val])
        elif (col_h1['변수']==i).any():
            t_stat, p_val = stats.ttest_ind(df[df['label']==1][i], df[df['label']==0][i], equal_var=False) # 이분산성 : welch’s t-test
            list.append([i, p_val])

    list = pd.DataFrame(list, columns = ['변수', 'p_value'])
    a = list[(list['p_value'] < p_value)][['변수', 'p_value']].sort_values('p_value')
    return a

# 2) t_test 결과 p_value < 0.05보다 작은 유의한 변수 가져오기
x_ttest = t_test(df.columns, x_homo, x_hetero, p_value=0.1)
print("유의한 피쳐 수 :", len(x_ttest))
x_ttest.sort_values(by="변수", ascending=True)
ttest = list(x_ttest['변수'])

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.simplefilter('ignore')

lasso_model = LogisticRegression()
param_grid = {'penalty' : ['l1'], 
                'C' : [0.001, 0.01, 0.1, 1, 2, 5, 10],
                'solver' : ['liblinear']}

grid_search = GridSearchCV(lasso_model, param_grid=param_grid, return_train_score=True, cv=5)
grid_search.fit(X_train, y_train)

df = pd.DataFrame(grid_search.cv_results_)
df = df.sort_values(by=['rank_test_score'], ascending=True)
df[['params', 'mean_train_score', 'mean_test_score', 'rank_test_score']]
print('GridSearchCV 최적 파라미터:', grid_search.best_params_)
print('GridSearchCV 최고 정확도:{0:.4f}'.format(grid_search.best_score_))

lasso_best = LogisticRegression(C=2, penalty='l1', solver='liblinear').fit(X_train, y_train)

df_lasso = pd.DataFrame()
df_lasso['feature'] = X_train.columns
df_lasso['coef'] = lasso_best.coef_[0]
df_lasso.drop(df_lasso[df_lasso['coef']==0].index, inplace=True)
df_lasso

# 라쏘에서 선택된 피처
lasso = df_lasso['feature'].values.tolist()
print('Lasso에서 선택된 피처 수 {0:1.0f}'.format(len(df_lasso)), '개')
lasso

list_Forward = list(Forward)
list_Backward = list(Backward)
list_Stepwise = list(Stepwise)
list_ttest= list(ttest)
list_lasso = list(lasso)
list_col_all = X_train.columns

def func_Forward(x):
    if x in list_Forward:
        return 1
    else:
        return 0
    
def func_Backward(x):
    if x in list_Backward:
        return 1
    else:
        return 0


def func_Stepwise(x):
    if x in list_Stepwise:
        return 1
    else:
        return 0

def func_ttest(x):
    if x in list_ttest:
        return 1
    else:
        return 0
    
def func_lasso(x):
    if x in list_lasso:
        return 1
    else:
        return 0

# 3번 선택된 Feature

feature_counts = pd.DataFrame()
feature_counts['Feature'] = list_col_all
feature_counts['Forward'] = list_col_all.map(func_Forward)
feature_counts['Backward'] = list_col_all.map(func_Backward)
feature_counts['Stepwise'] = list_col_all.map(func_Stepwise)
feature_counts['ttest'] = list_col_all.map(func_ttest)
feature_counts['lasso'] = list_col_all.map(func_lasso)

feature_counts["total"] = feature_counts["Forward"]+feature_counts['Backward']+feature_counts["Stepwise"]+feature_counts["ttest"]+feature_counts["lasso"]
feature_final = feature_counts[feature_counts["total"]>=3]
list_feature_final = list(feature_final["Feature"])
print("선택된 피쳐수 :", len(list_feature_final))

features = feature_final['Feature'].tolist()