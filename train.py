import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    import chardet
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'chardet'])
    import chardet


def detect_encoding(filename):
    with open(filename, 'rb') as f:
        raw = f.read()
    result = chardet.detect(raw)
    return result['encoding']


def read_csv_auto(filename):
    encodings = []
    try:
        encodings.append(detect_encoding(filename))
    except:
        pass
    encodings += ['cp949', 'euc-kr', 'utf-8-sig', 'utf-8', 'latin1']

    tried = []
    for enc in encodings:
        if not enc or enc in tried:
            continue
        tried.append(enc)
        try:
            return pd.read_csv(filename, encoding=enc)
        except:
            pass
    raise UnicodeDecodeError(f"{filename}를 읽을 수 없습니다.")


def standardize_date(date_str):
    s = str(date_str).strip()
    s = s.replace('년', '-').replace('월', '-').replace('일', '').replace(' ', '')
    parts = [p for p in s.split('-') if p]
    if len(parts) >= 2:
        year = parts[0]
        month = parts[1].zfill(2)
        day = parts[2].zfill(2) if len(parts) >= 3 else '01'
        return f"{year}-{month}-{day}"
    return s


def days_in_month(year, month):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    if month in [4, 6, 9, 11]:
        return 30
    if month == 2:
        leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        return 29 if leap else 28
    return 30


print("=== CSV 파일 읽기 ===")
weather_df = read_csv_auto('날씨.csv')
donor_df = read_csv_auto('헌혈자.csv')

print("날씨 데이터:", weather_df.shape)
print("헌혈자 데이터:", donor_df.shape)

print("\n=== 헌혈자 데이터 변환 (월별 -> 일일 평균) ===")
date_col = '날짜'
region_cols = [c for c in donor_df.columns if c != date_col]

donor_long = donor_df.melt(
    id_vars=[date_col],
    value_vars=region_cols,
    var_name='지역',
    value_name='월별헌혈자수'
)

donor_long['월별헌혈자수'] = (
    donor_long['월별헌혈자수']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.strip()
)

donor_long['월별헌혈자수'] = pd.to_numeric(donor_long['월별헌혈자수'], errors='coerce')
donor_long = donor_long.dropna(subset=['월별헌혈자수'])

def monthly_to_daily(row):
    s = str(row['날짜']).replace(' ', '')
    year = int(s.split('년')[0])
    month = int(s.split('년')[1].split('월')[0])
    return row['월별헌혈자수'] / days_in_month(year, month)

donor_long['헌혈자수'] = donor_long.apply(monthly_to_daily, axis=1)
donor_long['표준날짜'] = donor_long['날짜'].apply(standardize_date)

print(donor_long.head())

print("\n=== 날씨 날짜 표준화 ===")
weather_df['표준날짜'] = weather_df['날짜'].apply(standardize_date)
weather_df['강수량']=weather_df['강수량']/30

print("\n=== 데이터 병합 ===")
merged_data = pd.merge(
    weather_df,
    donor_long[['표준날짜', '지역', '헌혈자수']],
    on=['표준날짜', '지역'],
    how='inner'
)

print("병합 데이터 크기:", merged_data.shape)
print(merged_data.head())

print("\n=== merged_data 저장 ===")
merged_data.to_csv('merged_data.csv', index=False, encoding='utf-8-sig')

print("\n=== 모델 학습용 데이터 준비 ===")
feature_cols = ['평균기온', '최저기온', '최고기온', '강수량', '지역']
X = merged_data[feature_cols].copy()
y = merged_data['헌혈자수'].astype(float)

numeric_features = ['평균기온', '최저기온', '최고기온', '강수량']
categorical_features = ['지역']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

print("\n=== 예측 정확도 ===")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2 스코어: {r2:.4f} ({r2*100:.2f}%)")

print("\n=== 모델 저장 ===")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

model_info = {
    '모델': 'RandomForestRegressor',
    'MSE': mse,
    'RMSE': rmse,
    'MAE': mae,
    'R2': r2,
    'R2_%': r2 * 100,
    '학습데이터행수': len(merged_data)
}

with open('./blood_donor_model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

pd.DataFrame([model_info]).to_csv('모델_정확도_결과.csv', index=False, encoding='utf-8-sig')

def predict_donor_count(average_temp, min_temp, max_temp, precipitation, region):
    input_df = pd.DataFrame({
        '평균기온': [average_temp],
   '최저기온': [min_temp],
        '최고기온': [max_temp],
        '강수량': [precipitation],
        '지역': [region]
    })
    return model.predict(input_df)[0]

print("\n=== 예시 예측 ===")
print("서울:", round(predict_donor_count(15.0, 8.0, 22.0, 0.0, '서울'), 1))
print("부산:", round(predict_donor_count(20.0, 12.0, 28.0, 1.0, '부산'), 1))
print("대구:", round(predict_donor_count(25.0, 18.0, 32.0, 0.0, '대구'), 1))

print("\n완료: 헌혈자예측모델.pkl, merged_data.csv, 모델_정보.pkl 생성됨")