from sklearn.ensemble import IsolationForest
import pandas as pd


def remove_outliers(df: pd.DataFrame):
    tmp = df.copy(deep=True)
    tmp.drop(columns='property_id', inplace=True)

    isofor = IsolationForest(n_estimators=10000, bootstrap=True,
                             n_jobs=-1, random_state=1)

    outlier = isofor.fit_predict(df)

    return df[outlier >= 0]  # >= 0 implies that IsolationForest flags it as a non-outlier


