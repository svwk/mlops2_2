#!/usr/bin/env python
from clearml import Task, PipelineDecorator


@PipelineDecorator.component(cache=True, execution_queue="default")
def lib():
    pass


@PipelineDecorator.component(return_values=['df_train, df_test'], execution_queue="default")
def step_one(path_base):
    import pandas as pd
    df_train = pd.read_csv(path_base + '/train.csv', delimiter=',')
    df_test = pd.read_csv(path_base + '/test.csv', delimiter=',')

    return df_train, df_test


@PipelineDecorator.component(return_values=['X_train, X_valid, y_train, y_valid, X_test'], execution_queue="default")
def step_two(df_train, df_test, ):
    from sklearn.preprocessing import MinMaxScaler
    from data_methods import process_data
    from sklearn.model_selection import train_test_split

    min_max_scaler = MinMaxScaler()
    train_df_pr = process_data(df_train, min_max_scaler, True)
    test_df_pr = process_data(df_test, min_max_scaler, False)

    train_df_x = train_df_pr.drop(columns=["num_sold", "row_id"])
    train_df_y = train_df_pr["num_sold"].values
    X_train, X_valid, y_train, y_valid = train_test_split(train_df_x, train_df_y, test_size=0.3, random_state=42)
    X_test = test_df_pr.drop(columns="row_id")

    return X_train, X_valid, y_train, y_valid, X_test


@PipelineDecorator.component(return_values=['model, X_train, X_valid, X_test'], execution_queue="default")
def step_three(method, params, X_train, X_valid, X_test, y_train):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    from catboost import CatBoostRegressor

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
        model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            max_depth=params["max_depth"],
            reg_lambda=params["reg_lambda"]
        )
    elif method == 'CatBoostRegressor':
        model = CatBoostRegressor(
            iterations=params["n_estimators"],
            loss_function=params["loss_function"],
            learning_rate=params["learning_rate"],
            random_state=42,
            reg_lambda=params["reg_lambda"],
            max_depth=params["max_depth"]
        )
    model.fit(X_train, y_train)

    return model, X_train, X_valid, X_test


@PipelineDecorator.component(return_values=['smape_value'], execution_queue="default")
def step_four(model, X_valid, y_valid):
    from pmdarima.metrics import smape

    y_valid_pred = model.predict(X_valid)
    smape_value = smape(y_valid, y_valid_pred)

    return smape_value


@PipelineDecorator.component(return_values=['result_df'], execution_queue="default")
def step_five(model, X_test, df_test, path_base):
    import pandas as pd

    y_test_pred = model.predict(X_test)
    result_df = pd.concat([df_test['row_id'], pd.DataFrame(y_test_pred, columns=['num_sold'])], axis=1)
    result_df.num_sold = result_df.num_sold.astype('int')
    result_df.to_csv(path_base + "/submission.csv", index=False)

    return result_df


@PipelineDecorator.pipeline(name='TPS - Jan 2022-p', project='mlops2_2_pipeline', version='0.0.1')
def executing_pipeline():
    use_clearml = False
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
        "n_estimators": 120,
        "learning_rate": 0.1,
        "subsample": 0.5,
        "colsample_bytree": 1,
        "max_depth": 4,
        "gpu_id": -1,
        "loss_function": 'RMSE',
        "reg_lambda": 15
    }

    lib()
    task = Task.current_task()
    task.connect(params)
    logger = None
    if use_clearml:
        logger = task.get_logger()

    path_base = 'data'

    df_train, df_test = step_one(path_base)
    X_train, X_valid, y_train, y_valid, X_test = step_two(df_train, df_test)
    model, X_train, X_valid, X_test = step_three(method, params, X_train, X_valid, X_test, y_train)
    smape_value = step_four(model, X_valid, y_valid)
    result_df = step_five(model, X_test, df_test, path_base)
    print(smape_value)

    if use_clearml:
        logger.report_single_value(value=smape_value, name="smape")
        task.upload_artifact(name="submission", artifact_object=result_df)
        task.upload_artifact(name="model", artifact_object=model)
        task.mark_stopped(status_message="Done")


if __name__ == '__main__':
    PipelineDecorator.run_locally()
    executing_pipeline()
    PipelineDecorator.stop()
