import numpy as np
from doubleml.datasets import make_ssm_data
import doubleml as dml
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingClassifier, StackingRegressor
from xgboost import XGBRegressor, XGBClassifier
from DGP_functions import generate_dgp_1_MAR, generate_dgp_2_MAR, generate_dgp_3_MAR, generate_dgp_1_MNAR, generate_dgp_2_MNAR, generate_dgp_3_MNAR, generate_dgp_4_MNAR, generate_dgp_5_MNAR, generate_dgp_6_MNAR, generate_dgp_7_MNAR
from oracle_functions import prepare_oracle_dgp1_MAR, prepare_oracle_dgp2_MAR, prepare_oracle_dgp3_MAR, prepare_oracle_dgp1_MNAR, prepare_oracle_dgp2_MNAR, prepare_oracle_dgp3_MNAR, prepare_oracle_dgp4_MNAR, prepare_oracle_dgp5_MNAR, prepare_oracle_dgp6_MNAR, prepare_oracle_dgp7_MNAR
from doubleml.utils import DoubleMLResampling

dgp_functions_MAR = {
    1: generate_dgp_1_MAR,
    2: generate_dgp_2_MAR,
    3: generate_dgp_3_MAR,
}

dgp_functions_MNAR = {
    1: generate_dgp_1_MNAR,
    2: generate_dgp_2_MNAR,
    3: generate_dgp_3_MNAR,
    4: generate_dgp_4_MNAR,
    5: generate_dgp_5_MNAR,
    6: generate_dgp_6_MNAR,
    7: generate_dgp_7_MNAR
}

oracle_functions_MAR = {
    1: prepare_oracle_dgp1_MAR,
    2: prepare_oracle_dgp2_MAR,
    3: prepare_oracle_dgp3_MAR
}

oracle_functions_MNAR = {
    1: prepare_oracle_dgp1_MNAR,
    2: prepare_oracle_dgp2_MNAR,
    3: prepare_oracle_dgp3_MNAR,
    4: prepare_oracle_dgp4_MNAR,
    5: prepare_oracle_dgp5_MNAR,
    6: prepare_oracle_dgp6_MNAR,
    7: prepare_oracle_dgp7_MNAR
}

def perform_DML(dml_data, ml_g, ml_m, ml_pi, mar, y_real, grid_g, grid_m, grid_pi, tune, on_folds=False):
    
    # tune is necessary now
    s = dml_data.data["s"]
    y = dml_data.data["y"]
    d = dml_data.data["d"]

    results = []
    if mar:
        dml_ssm = dml.DoubleMLSSM(obj_dml_data=dml_data, ml_g_d0 = ml_g, ml_g_d1 = ml_g, ml_m = ml_m, ml_pi = ml_pi, score='missing-at-random')
        if tune:
            dml_ssm.tune({"ml_g_d0": grid_g, "ml_g_d1": grid_g, "ml_m": grid_m, "ml_pi": grid_pi}, tune_on_folds=on_folds, set_as_params=True)
    else: 
        dml_ssm = dml.DoubleMLSSM(obj_dml_data=dml_data, ml_g_d0=ml_g, ml_g_d1=ml_g, ml_m=ml_m, ml_pi=ml_pi, score='nonignorable')
        if tune:
            tune_res = dml_ssm.tune({"ml_g_d0": grid_g, "ml_g_d1": grid_g, "ml_m": grid_m, "ml_pi": grid_pi}, tune_on_folds=on_folds, return_tune_res=True)

    dml_ssm.fit(store_predictions=True)

    results.append(dml_ssm.summary["coef"].values[0])
    results.append(dml_ssm.summary["2.5 %"].values[0])
    results.append(dml_ssm.summary["97.5 %"].values[0])
    ml_g_pred = d * dml_ssm.predictions["ml_g_d1"][:,0,0] + (1 - d) * dml_ssm.predictions["ml_g_d0"][:,0,0] 

    rmse_ml_g = mean_squared_error(y_real, ml_g_pred)
    results.append(rmse_ml_g)

    logloss_ml_m = log_loss(d, dml_ssm.predictions["ml_m"][:,0,0])
    results.append(logloss_ml_m)

    logloss_ml_pi = log_loss(s, dml_ssm.predictions["ml_pi"][:,0,0])
    results.append(logloss_ml_pi)
    
    return pd.DataFrame([results], columns=["theta_hat", "2.5 %", "97.5 %", "ml_g_mse", "ml_m_log_loss", "ml_pi_log_loss"])

def simulate_oracle_one(dml_data, y_real, g_d0_oracle, g_d1_oracle, m_oracle, pi_oracle):

    y, d, s = dml_data.data["y"], dml_data.data["d"], dml_data.data["s"]

    dtreat = (d==1)
    dcontrol = (d==0)

    psi_b1 = (dtreat * s * (y - g_d1_oracle)) / (m_oracle * pi_oracle) + g_d1_oracle
    psi_b0 = (dcontrol * s * (y - g_d0_oracle)) / ((1 - m_oracle) * pi_oracle) + g_d0_oracle

    psi_diff = psi_b1 - psi_b0
    std_bias = np.std(psi_diff, ddof=1)
    se_bias = std_bias / np.sqrt(len(psi_diff))

    mean_diff = np.mean(psi_diff)

    z_value = 1.96  # for alpha 0.05%

    lower_bound = mean_diff - z_value * se_bias
    upper_bound = mean_diff + z_value * se_bias

    results = []

    results.append(mean_diff)
    results.append(lower_bound)
    results.append(upper_bound)

    ml_g_pred = d * g_d1_oracle + (1 - d) * g_d0_oracle 

    rmse_ml_g = mean_squared_error(y_real, ml_g_pred)
    results.append(rmse_ml_g)

    rmse_ml_m = log_loss(d, m_oracle)
    results.append(rmse_ml_m)

    rmse_ml_pi = log_loss(s, pi_oracle)
    results.append(rmse_ml_pi)


    return pd.DataFrame([results], columns=["theta_hat", "2.5 %", "97.5 %", "ml_g_mse", "ml_m_log_loss", "ml_pi_log_loss"])

def simulate_one(dml_data, mar, ml_model, y_real, tune, tuning_method):

    if ml_model == 'lasso':
        grid_g = {'alpha': np.logspace(-4, 1, 20)}
        grid_m = {'C': np.logspace(-4, 4, 10)}
        grid_pi = {'C': np.logspace(-4, 4, 10)}

        ml_g = Lasso()
        ml_m = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
        ml_pi = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)

    elif ml_model == 'rf':
        grid = {
            'max_depth': [5, 10],
            'n_estimators': [100, 300],
            'min_samples_leaf': [1, 5]
        }
        grid_g, grid_m, grid_pi = grid, grid, grid

        ml_g = RandomForestRegressor()
        ml_m = RandomForestClassifier()
        ml_pi = RandomForestClassifier()

    elif ml_model == 'xgb':

        grid_XGB = {
            'max_depth': [4, 5, 6],
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'n_jobs': [-1]
        }

        grid_g, grid_m, grid_pi = grid_XGB, grid_XGB, grid_XGB

        ml_g = XGBRegressor()
        ml_m = XGBClassifier(eval_metric='logloss')
        ml_pi = XGBClassifier(eval_metric='logloss')

    elif ml_model == 'regression':
        
        ml_g = LinearRegression()
        ml_m = LogisticRegression()
        ml_pi = LogisticRegression()
        grid_g = None
        grid_m = None
        grid_pi = None

    return perform_DML(dml_data=dml_data, ml_g=ml_g, ml_m=ml_m, ml_pi=ml_pi, mar=mar, y_real = y_real, grid_g = grid_g, grid_m = grid_m, grid_pi=grid_pi, tune=tune, on_folds = tuning_method == 'on_folds')
            
def simulate_one_for_all_models(seed, oracle, mar, ml_models, dgp_num, tune, tuning_method, n_obs=3000):
    
    if mar: 
        df = dgp_functions_MAR[dgp_num](seed, n_obs=n_obs)
        dml_data = dml.DoubleMLData(df.drop(columns='y_real'), 'y', 'd', s_col='s')
        __oracle_functions = oracle_functions_MAR[dgp_num](X = dml_data.data.drop(columns=["y","d","s"]), d = dml_data.data["d"])
        y_real = df["y_real"]
           
    else:
        df = dgp_functions_MNAR[dgp_num](seed, n_obs=n_obs)
        dml_data = dml.DoubleMLData(df.drop(columns='y_real'), 'y', 'd', s_col='s', z_cols='z')
        __oracle_functions = oracle_functions_MNAR[dgp_num](X = dml_data.data.drop(columns=["y","d","s","z"]), d = dml_data.data["d"], z = dml_data.data["z"], s= dml_data.data["s"])
        y_real = df["y_real"]

    results_from_all_ml_models = []

    for ml_model in ml_models:
        df_temp = simulate_one(dml_data = dml_data, mar=mar, ml_model=ml_model, y_real = y_real, tune=tune, tuning_method = tuning_method)
        df_temp['model'] = ml_model
        results_from_all_ml_models.append(df_temp)

    if oracle:
        df_temp = simulate_oracle_one(dml_data,y_real,__oracle_functions[0], __oracle_functions[1], __oracle_functions[2], __oracle_functions[3])
        df_temp['model'] = 'oracle'
        results_from_all_ml_models.append(df_temp)

    return pd.concat(results_from_all_ml_models)