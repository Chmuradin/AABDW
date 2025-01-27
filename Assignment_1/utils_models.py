xgb_params = {'n_estimators': [100, 200, 300],
              'max_depth': [3, 5, 7, 8],
              'min_child_weight': [3, 5],
              'gamma': [0.3, 0.5],
              'eta': [0.03, 0.05, 0.07, 0.1],
              'subsample': [0.8, 0.6],
              'colsample_bytree': [0.2, 0.1],
              'lambda': [1, 1.5, 2, 3],
              'alpha': [0,  0.1, 0.5, 1],
              'booster': ['dart']}


rf_params = {'n_estimators': [100, 200, 300,400, 500],
             'max_depth': [10, 12, 5, 6, 7, None],
             'max_features': [1.0, 'sqrt'],
             'min_samples_leaf': [1, 2, 4],
             'min_samples_split': [2, 5, 10],
             'bootstrap': [True, False]}





params = [xgb_params, rf_params]



