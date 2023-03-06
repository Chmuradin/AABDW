import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score


# Function to count the number of words in a string
def countWords(s: str) -> int:
    if s.strip() == "":
        return 0
    words = s.split()
    return len(words)


def data_wrangling(url: str, rare_list=[], cols=['property_id', 'property_lat', 'property_lon', 'property_type',
                                                 'property_room_type', 'property_max_guests',
                                                 'property_amenities', 'host_verified', 'booking_availability_365',
                                                 'reviews_num', 'reviews_rating', 'target']):
    # to be tested - are there any data leakages?
    data = pd.read_csv(url)
    if url == 'test.csv':
        data['target'] = np.nan
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


# Create pipelines
pipe_xgb = Pipeline([('scaler_xgb', StandardScaler()), ('pca_xgb', PCA()), ('xgb', XGBRegressor())])
pipe_log = Pipeline([('scaler_log', StandardScaler()), ('pca_log', PCA()), ('logistic', LogisticRegression())])
pipe_rf = Pipeline([('scaler_rf', StandardScaler()), ('pca_rf', PCA()), ('rf', RandomForestRegressor())])

# Create parameter grids
param_grid_xgb = {'xgb__max_depth': [2, 3],
                  'xgb__n_estimators': [100, 200,500,1000],
                  'xgb__colsample_bytree': [0.2,0.1],
                  'xgb__min_child_weight': [3, 5, 7],
                  'xgb__gamma': [0.3, 0.5, 0.7],
                  'xgb__subsample': [0.4,0.6,0.8]}
param_grid_log = {"pca_log__n_components": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                  "logistic__C": np.logspace(-4, 4, 8), }
param_grid_rf = {'rf__bootstrap': [True, False],
                 'rf__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                 'rf__max_features': [1.0, 'sqrt'],
                 'rf__min_samples_leaf': [1, 2, 4],
                 'rf__min_samples_split': [2, 5, 10],
                 'rf__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

# Load the data
data, rare_list = data_wrangling('train.csv')
final_testing_data = data_wrangling('test.csv', rare_list)[0].drop('target', axis=1)

X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1),
                                                    data['target'], test_size=0.2, random_state=42)

#search = GridSearchCV(pipe_log, param_grid_log, n_jobs=-1)
#search.fit(X_train, y_train)
#print("Best parameter (CV score=%0.3f):" % search.best_score_)
#print(search.best_params_)

search_xgb = RandomizedSearchCV(pipe_xgb, param_grid_xgb, n_jobs=-1, n_iter=150,verbose=2)
search_xgb.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search_xgb.best_score_)
print(search_xgb.best_params_)
fin_xgb = pd.DataFrame(search_xgb.predict(final_testing_data))
final_xgb = pd.concat([final_testing_data.property_id, fin_xgb], axis=1)

with open('out_xgb2.csv', 'w', newline='') as csv_file:
    final_xgb.to_csv(path_or_buf=csv_file, index=False, header=False)



#search_rf = RandomizedSearchCV(pipe_rf, param_grid_rf, n_jobs=2,n_iter=50,verbose=2)
#search_rf.fit(X_train, y_train)
#print("Best parameter (CV score=%0.3f):" % search_rf.best_score_)
#print(search_rf.best_params_)
# 'xgb__subsample': 0.8, 'xgb__n_estimators': 100, 'xgb__min_child_weight': 3, 'xgb__max_depth': 2, 'xgb__gamma': 0.3, 'xgb__colsample_bytree': 0.1
#tr_fin_rf = search_xgb.predict(final_testing_data)

#with open('out_rf2.csv', 'w', newline='') as csv_file:
 #   tr_fin_rf.to_csv(path_or_buf=csv_file, index=False, header=False)