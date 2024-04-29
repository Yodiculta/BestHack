import numpy as np
import pandas as pd
import streamlit as st
#py -m pip install streamlit
#py -m pip install numpy
#py -m pip install pandas
#py -m pip install joblib
# from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import pickle
import base64

def Xtransformer(xx):
    xx = xx.sort_values(['wagnum', 'ts_id', 'milleage_all']).bfill()
    xx = xx.drop_duplicates()
    xx['KP1'] = xx['axl1_l_w_flange'] +xx['axl1_r_w_flange']
    xx['KP2'] = xx['axl2_l_w_flange'] +xx['axl2_r_w_flange']
    xx['KP3'] = xx['axl3_l_w_flange'] +xx['axl3_r_w_flange']
    xx['KP4'] = xx['axl4_l_w_flange'] +xx['axl4_r_w_flange']
    grouped = xx.groupby(['wagnum', 'ts_id']).agg(
        Max_probeg=('milleage_all', 'max'),
        Avg_probeg=('milleage_all', 'mean'),
        Min_probeg=('milleage_all', 'min'),
        axl1_l_w_flange_diff=('axl1_l_w_flange', lambda x: x.iloc[0] - x.iloc[-1]),
        axl1_r_w_flange_diff=('axl1_r_w_flange', lambda x: x.iloc[0] - x.iloc[-1]),
        axl2_l_w_flange_diff=('axl2_l_w_flange', lambda x: x.iloc[0] - x.iloc[-1]),
        axl2_r_w_flange_diff=('axl2_r_w_flange', lambda x: x.iloc[0] - x.iloc[-1]),
        axl3_l_w_flange_diff=('axl3_l_w_flange', lambda x: x.iloc[0] - x.iloc[-1]),
        axl3_r_w_flange_diff=('axl3_r_w_flange', lambda x: x.iloc[0] - x.iloc[-1]),
        axl4_l_w_flange_diff=('axl4_l_w_flange', lambda x: x.iloc[0] - x.iloc[-1]),
        axl4_r_w_flange_diff=('axl4_r_w_flange', lambda x: x.iloc[0] - x.iloc[-1]),
        axl1_l_w_flange_min=('axl1_l_w_flange', 'min'),
        axl1_r_w_flange_min=('axl1_r_w_flange', 'min'),
        axl2_l_w_flange_min=('axl2_l_w_flange', 'min'),
        axl2_r_w_flange_min=('axl2_r_w_flange', 'min'),
        axl3_l_w_flange_min=('axl3_l_w_flange', 'min'),
        axl3_r_w_flange_min=('axl3_r_w_flange', 'min'),
        axl4_l_w_flange_min=('axl4_l_w_flange', 'min'),
        axl4_r_w_flange_min=('axl4_r_w_flange', 'min'),
        axl1_l_w_flange_mean=('axl1_l_w_flange', 'mean'),
        axl1_r_w_flange_mean=('axl1_r_w_flange', 'mean'),
        axl2_l_w_flange_mean=('axl2_l_w_flange', 'mean'),
        axl2_r_w_flange_mean=('axl2_r_w_flange', 'mean'),
        axl3_l_w_flange_mean=('axl3_l_w_flange', 'mean'),
        axl3_r_w_flange_mean=('axl3_r_w_flange', 'mean'),
        axl4_l_w_flange_mean=('axl4_l_w_flange', 'mean'),
        axl4_r_w_flange_mean=('axl4_r_w_flange', 'mean'),
        KP1_max = ('KP1', 'max'),
        KP2_max = ('KP2', 'max'),
        KP3_max = ('KP3', 'max'),
        KP4_max = ('KP4', 'max')        
    )
    grouped = grouped.reset_index()
    return grouped
    

def post_transformer(xx, y):
    groupedx = Xtransformer(xx)
    merged_df = groupedx.merge(y, on=['wagnum', 'ts_id'] , how = 'inner', suffixes=('_target', ''))
    X_train = merged_df.drop(['target'],  axis=1)
    y_train = merged_df['target']
    return X_train, y_train

def modeling(X_t, y_t):
    train_data_new = pd.read_parquet('train.parquet')
    train_target = pd.read_csv('train_target.csv')
    train_data_new = train_data_new.sort_values(['wagnum', 'ts_id', 'milleage_all']).bfill()
    train_data_new = train_data_new.drop_duplicates()
    X_train,y_train = post_transformer(train_data_new, train_target)
    X_test, y_test = post_transformer(X_t, y_t)
    
    pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('regressor', KNeighborsRegressor())#iterations=2000, learning_rate=0.01, depth=11, loss_function='MAE', random_seed=42))
    ])  
    pipeline.fit(X_train,y_train)
    pipeline.score(X_train,y_train)
    y_pred = pipeline.predict(X_test)
    return mean_absolute_error(y_test, y_pred)#, y_pred

def predictiong(X_t):
    train_data_new = pd.read_parquet('train.parquet')
    train_target = pd.read_csv('train_target.csv')
    train_data_new = train_data_new.sort_values(['wagnum', 'ts_id', 'milleage_all']).bfill()
    train_data_new = train_data_new.drop_duplicates()
    X_train,y_train = post_transformer(train_data_new, train_target)
    X_test = Xtransformer(X_t)
    
    pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('regressor', KNeighborsRegressor())#iterations=2000, learning_rate=0.01, depth=11, loss_function='MAE', random_seed=42))
    ])  
    pipeline.fit(X_train,y_train)
    pipeline.score(X_train,y_train)
    y_pred = pipeline.predict(X_test)
    print(y_pred)
    return y_pred

st.title("Best Hack 2023*")
st.text("Produced by: Iriski")
st.write("* ВНИМНИЕ! В демонтрации интерфейса использовалось не конечное целевое решение, т.к. из-з а технических проблем, не удалоось установить в окружение проекта CatBoostRegressor")
uploaded_file = st.file_uploader("Choose a file with answers (parquet)")
uploaded_file_y = st.file_uploader("Choose a file with data (csv)")

def predict():
    y_test = None
    X_test = None
    if uploaded_file is not None:
        X_test = pd.read_parquet(uploaded_file)
        
    if uploaded_file_y is not None:
        y_test = pd.read_csv(uploaded_file_y)
    if (y_test is not None)&(X_test is not None):
            st.write("mean absolure error:")
            st.write(modeling(X_test, y_test))
            


st.button('Predict', on_click = predict)

    


uploaded_file2 = st.file_uploader("Choose a file to get prediction")

X_test = None
if uploaded_file2 is not None:
    X_test = pd.read_parquet(uploaded_file2)
    if (X_test is not None):
        st.write("y predicted:")
        st.write(predictiong(X_test))