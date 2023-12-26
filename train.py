#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from clearml import Task
from pmdarima.metrics import smape
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

from data_methods import process_data

use_clearml = True
show_plots = False
iteration = 1

# LinearRegression
# PolynomialRegression
# RandomForestRegressor
# XGBRegressor
# CatBoostRegressor
method = "CatBoostRegressor"

params = {
    "method": method,
    "n_estimators": 130,
    "learning_rate": 0.1,
    "subsample": 0.5,
    "colsample_bytree": 1,
    "max_depth": 4,
    "gpu_id": -1,
    "loss_function": 'RMSE',
    "reg_lambda": 15}

logger = None
task = None
if use_clearml:
    task = Task.init(project_name='mlops2_2', task_name='TPS - Jan 2022', reuse_last_task_id=False)
    task.connect(params)
    logger = task.get_logger()

PATH_base = 'data'
df_train = pd.read_csv(PATH_base + '/train.csv', delimiter=',')
df_test = pd.read_csv(PATH_base + '/test.csv', delimiter=',')
df_submit = pd.read_csv(PATH_base + '/sample_submission.csv', delimiter=',')

df_train["date"] = pd.to_datetime(df_train["date"])
df_test["date"] = pd.to_datetime(df_test["date"])

if show_plots:
    df1 = df_train.groupby([pd.Grouper(key="date", freq="1M"), 'store'])[["num_sold"]].sum()
    fig2 = plt.figure(figsize=(20, 10))
    barplot = sns.lineplot(data=df1, x='date', y='num_sold', hue='store', palette="tab10", linewidth=2.5)
    barplot.set_title('Динамика изменения объема продаж для разных магазинов', fontsize=16)
    barplot.set_xlabel('Период', fontsize=16);
    barplot.set_ylabel('Суммарные продажи', fontsize=16);
    barplot.grid()
    fig2.show()

    df1 = df_train.groupby([pd.Grouper(key="date", freq="1M"), 'product'])[["num_sold"]].sum()
    fig1 = plt.figure(figsize=(20, 10))
    barplot = sns.lineplot(data=df1, x='date', y='num_sold', hue='product', palette="tab10", linewidth=2.5)
    barplot.set_title('Динамика изменения объема продаж для разных товаров', fontsize=16)
    barplot.set_xlabel('Период', fontsize=16);
    barplot.set_ylabel('Суммарные продажи', fontsize=16);
    barplot.grid()
    fig1.show()

# Feature processing
min_max_scaler = MinMaxScaler()
train_df_pr = process_data(df_train, min_max_scaler, True)
test_df_pr = process_data(df_test, min_max_scaler, False)

train_df_x = train_df_pr.drop(columns=["num_sold", "row_id"])
train_df_y = train_df_pr["num_sold"].values
X_train, X_valid, y_train, y_valid = train_test_split(train_df_x, train_df_y, test_size=0.3, random_state=42)
X_test = test_df_pr.drop(columns="row_id")

# Обучение
model = LinearRegression()
if method == "PolynomialRegression":
    degree = 2
    poly = PolynomialFeatures(degree=degree)
    X_train = poly.fit_transform(X_train)
    X_valid = poly.transform(X_valid)
    X_test = poly.transform(X_test)
    model = LinearRegression()

elif method == "RandomForestRegressor":
    model = RandomForestRegressor(n_estimators=params["n_estimators"], random_state=42)
elif method == "XGBRegressor":
    model = XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=params["n_estimators"],
                         learning_rate=params["learning_rate"], subsample=params["subsample"],
                         colsample_bytree=params["colsample_bytree"], max_depth=params["max_depth"],
                         reg_lambda=params["reg_lambda"])
elif method == 'CatBoostRegressor':
    model = CatBoostRegressor(iterations=params["n_estimators"], loss_function=params["loss_function"],
                              learning_rate=params["learning_rate"], random_state=42, reg_lambda=params["reg_lambda"],
                              max_depth=params["max_depth"])

model.fit(X_train, y_train)
y_valid_pred = model.predict(X_valid)
score = model.score(X_valid, y_valid)
smape_value = smape(y_valid, y_valid_pred)

if use_clearml:
    logger.report_single_value(value=smape_value, name="smape")
    logger.report_single_value(value=score, name="score")
print(smape_value)

y_test_pred = model.predict(X_test)
result_df = pd.concat([df_test['row_id'], pd.DataFrame(y_test_pred, columns=['num_sold'])], axis=1)
result_df.num_sold = result_df.num_sold.astype('int')

# result_df.to_csv(PATH_base + "/submission.csv", index=False)

if use_clearml:
    task.upload_artifact(name="submission", artifact_object=result_df.values)
    task.upload_artifact(name="model", artifact_object=model)
    # task.mark_stopped(status_message="Done")
