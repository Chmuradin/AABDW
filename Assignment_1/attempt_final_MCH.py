import pandas as pd
from sklearn.model_selection import train_test_split,  RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from utils_models import params


def make_predictions(model, pred_data):
    pred = model.predict(pred_data)
    pred = pd.DataFrame(pred, columns=['target'])
    return pred


def find_best_model(regressor, param_grid,  data):
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['target'], axis=1), data['target'], test_size=0.2, random_state=42)
    model = RandomizedSearchCV(regressor, param_grid, n_iter=5, cv=5, n_jobs=-1, verbose=2)
    model.fit(X_train, y_train)
    print(model.best_params_)
    print(model.best_score_)
    print(model.score(X_test, y_test))
    return model


def main():
    data = pd.read_csv('train_transformed.csv')
    pred_data = pd.read_csv('test_transformed.csv')
    target = pd.read_csv('train.csv')
    data = pd.concat([data, target['target']], axis=1)
    all_preds = pd.DataFrame(pred_data['property_id'])
    models = [XGBRegressor(), RandomForestRegressor()]
    for model, param_grid in zip(models, params):
        model = find_best_model(model, param_grid, data)
        all_preds = pd.concat([all_preds, make_predictions(model, pred_data)], axis=1)
    print(all_preds.head())
    all_preds['fin_target'] = all_preds.drop(['property_id'], axis=1).mean(axis=1)
    print(all_preds.head())
    with open('submission905.csv', 'w') as f:
        f.write(all_preds[['property_id', 'fin_target']].to_csv(index=False))

def supp():
    data = pd.read_csv('train_transformed.csv')
    pred_data = pd.read_csv('test_transformed.csv')
    target = pd.read_csv('train.csv')
    data = pd.concat([data, target['target']], axis=1)
    all_preds = pd.DataFrame(pred_data['property_id'])
    par = {'subsample': 0.8, 'n_estimators': 100, 'min_child_weight': 5, 'max_depth': 3, 'lambda': 3, 'gamma': 0.3, 'eta': 0.07, 'colsample_bytree': 0.1, 'booster': 'dart', 'alpha': 0.5}
    model = XGBRegressor(**par).fit(data.drop(['target'], axis=1), data['target'])
    par = {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 5, 'bootstrap': True}
    model = RandomForestRegressor(**par).fit(data.drop(['target'], axis=1), data['target'])
    fin = pd.DataFrame(model.predict(pred_data))
    fin = pd.concat([all_preds, fin], axis=1)
    print(fin.head())
    with open('submission905_tmp2.csv', 'w') as f:
        f.write(fin.to_csv(index=False, header=False))

def kurwamac():
    data = pd.read_csv('submission905_tmp.csv', header=None)
    data.columns = ['f', 'fff']
    data2 = pd.read_csv('submission905_tmp2.csv', header=None)
    data2.columns = ['e', 'eee']

    all_preds = pd.concat([data, data2], axis = 1)
    
print(all_preds.head())
all_preds['fin_target'] = all_preds.drop(['e', 'f'], axis=1).mean(axis=1)
print(all_preds.head())
final = all_preds[['f','fin_target']]
print(final)
print()
with open('submission905_tmp12.csv', 'w') as f:
    f.write(final.to_csv(index=False, header=False))

if __name__ == '__main__':
    supp()