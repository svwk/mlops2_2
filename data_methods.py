import pandas as pd


def process_data(df_source, scaler, to_fit_scaler=False):
    """
    Предобработка данных
    :param df_source: исходный датасет
    :param scaler: объект масштабатора
    :param to_fit_scaler: флаг необходимости обучения масштабатора
    :return: обработанный датасет
    """
    df_source["date"] = pd.to_datetime(df_source["date"])
    df_one_hot_encoded = pd.get_dummies(df_source[['country', 'store', 'product']],
                                        prefix=['country', 'store', 'product'], dtype=int)
    df_new = pd.concat([df_source, df_one_hot_encoded], axis=1)
    df_new['year'] = df_new['date'].dt.year
    df_new['month'] = df_new['date'].dt.month
    df_new["day"] = df_new["date"].dt.day
    df_new["dayofweek"] = df_new["date"].dt.dayofweek

    columns_to_drop = ["date", "country", "store", "product"]
    df_new = df_new.drop(columns=columns_to_drop)

    col_modify_scaling = ["dayofweek", "day", "month"]

    if to_fit_scaler:
        scaler.fit(df_new[col_modify_scaling])

    df_new[col_modify_scaling] = scaler.transform(df_new[col_modify_scaling])
    df_new['year'] = df_new['year'].rank(method='dense').astype(int)

    return df_new
