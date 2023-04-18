import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso
from utils_models import params
import re


def data(train_url, comp_url):
    train_data, test_data = train_test_split(pd.read_csv(train_url), test_size=0.2, random_state=42)
    competition_data = pd.read_csv(comp_url)
    return train_data, test_data, competition_data


def param_searching(model, param_grid, train_data):
    # This function returns parameters for the given model
    if len(type(model).__name__) > 7:
        rsc = RandomizedSearchCV(model, param_grid, cv=5, n_iter=120, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    else:
        rsc = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    rsc_fit = rsc.fit(train_data.drop(['target', 'property_id'], axis=1), train_data['target'])
    res = rsc_fit.best_estimator_
    return res



def predictions_for_ensemble(models, test_data, comp_data):
    # This function takes a list of models and returns a list of predictions for each model.
    predictions_df = pd.DataFrame(columns=[type(model).__name__ for model in models])
    predictions_df['property_id'] = comp_data['property_id']
    scores = []
    for model in models:
        print(type(model).__name__)
        predictions = model.predict(comp_data.drop('property_id', axis=1))
        score = model.score(test_data.drop(['target', 'property_id'], axis=1), test_data['target'])
        # neg_mse = cross_val_score(model, train_data.drop(['target', 'property_id'], axis=1), train_data['target'], cv=2, scoring='neg_mean_squared_error', verbose=2).mean()
        predictions_df[type(model).__name__] = predictions
        scores.append({'value_'+str(type(model).__name__): score}) # type(model).__name__:
    return predictions_df, scores
    # scores on a train set, will have to test on a test set


def main():
    # Your code here
    train, val, comp = data('PCA_temp_train_data.csv', 'PCA_temp_test_data.csv')
    model_list = [XGBRegressor(), RandomForestRegressor(), Lasso(), Ridge()]
    best_model_list = [param_searching(model, param, train) for model, param in zip(model_list, params)]
    predictions_list, scores = predictions_for_ensemble(best_model_list, val, comp)
    predictions_list.to_csv('predictions.csv', index=False)
    print(predictions_list.head(10))
    print(scores)
    #weights = [float(d.values()) for d in scores] / sum([d.values() for d in scores])
    weights = [0.49, 0.49, 0.01, 0.01]
    ensemble_prediction = sum([predictions_list.iloc[:, i] * weights[i] for i in range(len(weights))])
    final_pred = pd.DataFrame({'property_id': predictions_list['property_id'], 'target': ensemble_prediction})
    final_pred.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    main()
