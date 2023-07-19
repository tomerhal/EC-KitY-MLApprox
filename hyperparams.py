import optuna
from optuna.samplers import TPESampler

from sklearn.linear_model import Ridge, Lasso, SGDRegressor, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

import pandas as pd
import os

import utils


def objective(trial, X, y, w):
    regressor_name = trial.suggest_categorical("regressor", ["Lasso", "Ridge", "SGDRegressor", "ElasticNet"])
    alpha = trial.suggest_float("alpha", 1e-5, 1e5, log=True)
    tol = trial.suggest_float("tol", 1e-10, 1e-4)
    max_iter = trial.suggest_int('max_iter', 1000, 10_000)
    if regressor_name == "Lasso":
        regressor_obj = Lasso(alpha=alpha, max_iter=max_iter, tol=tol)
    elif regressor_name == "Ridge":
        regressor_obj = Ridge(alpha=alpha, max_iter=max_iter, tol=tol)
    elif regressor_name == "SGDRegressor":
        eta0 = trial.suggest_float('eta0', 1e-3, 1e-1)
        learning_rate = trial.suggest_categorical('lr', ['constant', 'optimal', 'invscaling', 'adaptive'])
        penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'])
        l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
        early_stopping = trial.suggest_categorical('early_stopping', [True, False])
        validation_fraction = trial.suggest_categorical('val_frac', [0.1, 0.15, 0.2, 0.25, 0.3])
        regressor_obj = SGDRegressor(alpha=alpha, max_iter=max_iter, tol=tol, 
                                     eta0=eta0, learning_rate=learning_rate,
                                     penalty=penalty, l1_ratio=l1_ratio, 
                                     early_stopping=early_stopping, 
                                     validation_fraction=validation_fraction)
    elif regressor_name == "ElasticNet":
        l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
        regressor_obj = ElasticNet(alpha=alpha, max_iter=max_iter, tol=tol, l1_ratio=l1_ratio)
    else:
        exit('error: unknown regressor')

    fit_params = {"sample_weight": w}
    scoring = make_scorer(mean_squared_error)

    score = cross_val_score(regressor_obj, X, y,
                            n_jobs=-1, cv=5,
                            scoring=scoring,
                            fit_params=fit_params)
    error = score.mean()
    return error


if __name__ == "__main__":
    merged_df = None
    file_names = [file for file in os.listdir('datasets') if file.startswith('9') and file.endswith('.csv')]
    # job_ids = [str(i) for i in range(8607399, 8607408)]
    # file_names = [f'{job_id}.csv' for job_id in job_ids]

    for file in file_names:
        df = pd.read_csv('datasets/' + file)
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.concat([merged_df, df], ignore_index=True, copy=False)

    n_features = merged_df.shape[1] - 2

    merged_df.drop_duplicates(inplace=True)
    X, y = merged_df.iloc[:, :-2].to_numpy(), merged_df['fitness'].to_numpy()

    # sample weights
    w = merged_df['gen'].to_numpy()
    w = utils.square_gen_weight(w)

    print('X.shape', X.shape)
    print('y.shape', y.shape)

    direction = 'minimize'
    study = optuna.create_study(direction=direction, sampler=TPESampler())
    study.optimize(lambda trial: objective(trial, X, y, w), n_trials=5000, n_jobs=-1)
    print('best_trial\n', study.best_trial)
    print('best_params\n', study.best_params)
    print('best_value\n', study.best_value)
