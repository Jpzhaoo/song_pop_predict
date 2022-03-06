# %%
# 加载相关包
import pandas as pd
import numpy as np

from plotnine import *
from matplotlib import pyplot as plt
import seaborn as sns 

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


import optuna


import warnings
warnings.filterwarnings('ignore')



# %%
#------------------------------------
# 读取数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
df_test = test.copy()
submission = pd.read_csv("sample_submission.csv")
print(train.shape, test.shape, submission.shape)
#------------------------------------


# %%
#------------------------------------
# 不同变量查看
# features = train.drop(['id', 'song_popularity'], axis=1).columns.tolist()
features = [col for col in train.columns if col not in ['id', 'song_popularity']]
cat_var = ['key', 'audio_mode', 'time_signature']
cont_var = [i for i in features if i not in cat_var]




# %%
# 缺失值处理
#------------------------------------
missing_count = pd.DataFrame([train.isnull().mean(), test.isnull().mean()]).T.reset_index()
missing_count.columns = ['columns','train_missing', "test_missing"]
# (
#     ggplot(pd.melt(missing_count, id_vars='columns', 
#         value_vars=['train_missing', 'test_missing'],
#         var_name='variable', value_name='value')) +
#     geom_col(aes(x = 'columns', y = 'value', fill = 'variable'))
# )


plt.figure(figsize=(10,6))
missing_count \
    .query("train_missing > 0") \
    .plot(kind="barh",
          title="% of Missing Value",
          )
plt.show()



# 插值
# imputer_1 = SimpleImputer(strategy='median')
cat_imputer = KNNImputer(weights="distance")
cont_imputer = IterativeImputer()

train_cat_im = pd.DataFrame(cat_imputer.fit_transform(train[cat_var]))
train_cat_im.columns = cat_var
test_cat_im = pd.DataFrame(cat_imputer.transform(test[cat_var]))
test_cat_im.columns = cat_var

train_cont_im = pd.DataFrame(cont_imputer.fit_transform(train[cont_var]))
train_cont_im.columns = cont_var
test_cont_im = pd.DataFrame(cont_imputer.transform(test[cont_var]))
test_cont_im.columns = cont_var

train[cat_var] = train_cat_im
test[cat_var] = test_cat_im
train[cont_var] = train_cont_im 
test[cont_var] = test_cont_im

# train_im = pd.DataFrame(imputer_2.fit_transform(train[features]))
# train_im.columns = features
# test_im = pd.DataFrame(imputer_2.transform(test[features]))
# test_im.columns = features

# train[features] = train_im
# test[features] = test_im
# test = test[features]
#------------------------------------
 
# %%
#------------------------------------
# 转换
le = RobustScaler()
train[cont_var] = le.fit_transform(train[cont_var])
test[cont_var] = le.transform(test[cont_var])

# %%
#------------------------------------
# kfold
# train['kfold'] = -1
# kfold = KFold(n_splits=5, shuffle=True, random_state=423)

# for fold, (tr_id, val_id) in enumerate(kfold.split(X=train)):
#     train.loc[val_id, 'kfold'] = fold
    
# train.kfold.value_counts()

# train.to_csv('fold_5.csv', index=False)

# model
X = train.drop(['id', 'song_popularity'], axis=1)
y = train.song_popularity
test = test.drop(['id'], axis=1)

# 类别变量
cat_indices = []
for f in cat_var:
    if f in X.columns:
        idx = list(X.columns).index(f)
        cat_indices.append(idx)

def objective(trial, data=X, target=y):
    
    params = {
                'metric': 'auc', 
                'random_state': 22,
                'n_estimators': trial.suggest_categorical('n_estimators', [3000, 4000, 5000, 6000, 7000, 8000]),
                'boosting_type': trial.suggest_categorical("boosting_type", ["gbdt"]),
                'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-3, 10.0),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-3, 10.0),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                'bagging_fraction': trial.suggest_categorical('bagging_fraction', [0.6, 0.7, 0.80]),
                'feature_fraction': trial.suggest_categorical('feature_fraction', [0.6, 0.7, 0.80]),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
                'max_depth': trial.suggest_int('max_depth', 2, 12, step=1),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
                'num_leaves' : trial.suggest_int('num_leaves', 10, 200, step=20),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 96, step=5),
            }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=423)
    cv_scores = np.empty(5)
    for idx, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        clf = lgb.LGBMClassifier(**params)  
        clf.fit(X_train, y_train,
                eval_set=[(X_valid, y_valid), (X_train, y_train)],
                categorical_feature=cat_indices,
                callbacks=[
                       lgb.early_stopping(stopping_rounds=100)
                        ],
            )
        
        y_proba = clf.predict_proba(X_valid)[:, 1]
        cv_scores[idx] = roc_auc_score(y_valid, y_proba)
    return np.mean(cv_scores)

params_search = True
if params_search:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    
if params_search:
    print("\n")
    print("Best parameters:")
    print("="*30)
    for param, v in study.best_trial.params.items():
        print(f"{param} :\t {v}")
    print(f"Best AUC: {study.best_value}")
    
    
    
# %%
NFOLDS = 5

stratified_kfolds = StratifiedKFold(n_splits=NFOLDS)
stratified_kfolds.get_n_splits(X, y, groups=y)

if not type(X) == np.ndarray:
    X = X.values; y = y.values
    
fold = 1

preds = np.zeros((test.shape[0]))
for train_idx, val_idx in stratified_kfolds.split(X, y, groups=y):
    print(f"FOLD: {fold}")
    
    X_train, X_valid = X[train_idx], X[val_idx]
    y_train, y_valid = y[train_idx], y[val_idx]
    
    # further manually tuned params from Optuna tunes best params
    params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_estimators": 5000,
            "is_unbalance": "true",
            "max_depth": 3,
            "num_leaves": 130,
            "lambda_l1": 0.9575736933289517,
            "lambda_l2": 0.6037203927053586,
            "min_child_samples": 76,
            "learning_rate": 0.01698934395490724, 
            "min_gain_to_split": 0.275018712492725,
            "colsample_bytree": 0.5,
            "bagging_fraction": 0.7,
            "feature_fraction": 0.7,
            "metric": ["auc"],
            "verbose": -1
            }

    clf = lgb.LGBMClassifier(**params)  
    clf.fit(X_train, y_train,
            eval_set=[(X_valid, y_valid), (X_train, y_train)],
            categorical_feature=cat_indices,
            callbacks=[
                # lgb.log_evaluation(period=100), 
                       lgb.early_stopping(stopping_rounds=500)
                      ],
           )
    # test prediciton
    preds += clf.predict_proba(test)[:, 1]/NFOLDS
    # validation prediction
    y_proba = clf.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_proba)
    print("Validation AUC: ", auc)
    
    fold += 1
    
    
# %%
submission = pd.DataFrame(data={"id": df_test.id.values, "song_popularity": preds})
submission.to_csv("submission.csv", index=False)
# %%
