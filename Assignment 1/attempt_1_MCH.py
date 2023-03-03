import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


def train_val_test(data: pd.DataFrame):
    df, validation_df = train_test_split(data, test_size=0.01, random_state=42)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    return df_train, df_test, validation_df


def countWords(s: str) -> int:
    if s.strip() == "":
        return 0
    words = s.split()
    return len(words)


def data_wrangling(url: str, rare_list=[]):
    # to be tested - are there any data leakages?
    data = pd.read_csv(url)
    data = data[cols]
    data['reviews_rating'] = data['reviews_rating'].fillna(0)
    data['property_amenities'] = data['property_amenities'].fillna('').apply(lambda s: countWords(s))
    data['host_verified'] = data['host_verified'].apply(lambda s: countWords(s))
    if url == 'train.csv':
        rare = data['property_type'].value_counts()
        rare_list = list(rare[rare < 30].index)
    for typ in rare_list:
        data['property_type'] = data['property_type'].replace(typ, 'Other')
    data = pd.concat([data, pd.get_dummies(data['property_room_type'], prefix='property_room_type')], axis=1)
    data = pd.concat([data, pd.get_dummies(data['property_type'], prefix='property_type')], axis=1)
    data = data.drop(['property_type', 'property_room_type'], axis=1)
    return data, rare_list


def model_training(data: pd.DataFrame, params):
    df_train, df_test = train_test_split(data, test_size=0.2, random_state=42)
    X_train, y_train = df_train.drop('target', axis=1), df_train['target']
    X_test, y_test = df_test.drop('target', axis=1), df_test['target']
    model = xgb.XGBRegressor()
    rs = GridSearchCV(
    model, # model to optimize
    param_grid=params,
    scoring='neg_mean_squared_error', # metric to evaluate model performance
    cv=5, # cross-validation splitting strategy
    verbose=1, # controls verbosity of the search process
    n_jobs=-1 # number of CPU cores used for parallel computing
     )
    rs.fit(X_train, y_train)
    print(rs.best_params_, rs.best_score_)
    return rs, X_test, y_test

def random_forest_regress(data:pd.DataFrame, params):
    df_train, df_test = train_test_split(data, test_size=0.01, random_state=42)
    X_train, y_train = df_train.drop('target', axis=1), df_train['target']
    X_test, y_test = df_test.drop('target', axis=1), df_test['target']
    model = RandomForestRegressor()
    rfr = GridSearchCV(model, param_grid=params,
    scoring='neg_mean_squared_error', # metric to evaluate model performance
    cv=5, # cross-validation splitting strategy
    verbose=2, # controls verbosity of the search process
    n_jobs=-1 # number of CPU cores used for parallel computing
     )
    rfr.fit(X_train, y_train)
    print(rfr.best_params_, rfr.best_score_)
    return rfr, X_test, y_test


def score_test(model, X_test, y_test):
    print(mean_squared_error(y_test, model.predict(X_test)))


def preds(model):
    cols.remove('target')
    test_set, tmp = data_wrangling('test.csv', rare_list)
    y_pred = pd.DataFrame(model.predict(test_set))
    #y_pred_fin = pd.DataFrame(scal.inverse_transform(pd.concat([test_set, y_pred])))
    return pd.concat([test_set.property_id, y_pred], axis=1)


cols = ['property_id', 'property_lat', 'property_lon', 'property_type', 'property_room_type',
        'property_max_guests',
        'property_amenities', 'host_verified', 'booking_availability_365', 'reviews_num', 'reviews_rating', 'target']

param_dict = { 'max_depth': [2, 3],
           'n_estimators': [100, 200],
           'colsample_bytree': [0.2],
           'min_child_weight': [3, 5, 7],
           'gamma': [0.3, 0.5],
           'subsample': [0.8]}

param_RFR = {
    'n_estimators': [100, 200],
    'criterion': ['friedman_mse', 'squared_error'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [ 4],
    'max_features': ['log2']
}




tr_data, rare_list = data_wrangling('train.csv')
#scal = StandardScaler().fit(tr_data)
#scaled_data = pd.DataFrame(scal.transform(tr_data), columns=tr_data.columns)
#mod, x_test, y_test = model_training(tr_data, param_dict)
mod, x_test, y_test = random_forest_regress(tr_data, param_RFR)
#score_test(mod, x_test, y_test)
final = preds(mod)

with open('out_RFR.csv', 'w', newline='') as csv_file:
    final.to_csv(path_or_buf=csv_file, index=False, header=False)
