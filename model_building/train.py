import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV
import os
import joblib
from huggingface_hub import login, HfApi, create_repo
import xgboost as xgb


api = HfApi()

X_train_path = "hf://datasets/Amidho/Machine-Failure-Prediction/Xtrain.csv"
X_test_path = "hf://datasets/Amidho/Machine-Failure-Prediction/Xtest.csv"
y_train_path = "hf://datasets/Amidho/Machine-Failure-Prediction/ytrain.csv"
y_test_path = "hf://datasets/Amidho/Machine-Failure-Prediction/ytest.csv"

X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path)
y_test = pd.read_csv(y_test_path)


numeric_features = [
    'Air temperature',
    'Process temperature',
    'Rotational speed',
    'Torque',
    'Tool wear'
]

categorical_features = ['Type']

class_weight = y_train.value_counts()[0]/y_train.value_counts()[1]

preprocessor = make_column_transformer(
    (StandardScaler(),numeric_features),
    (OneHotEncoder(handle_unknown='ignore'),categorical_features)
)

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42
)


param_grid = {
    "xgbclassifier__n_estimators":[50,75,100],
    "xgbclassifier__max_depth":[3,4,5],
    "xgbclassifier__learning_rate":[0.01,0.02,0.03],
    "xgbclassifier__colsample_bytree":[0.4,0.5,0.6],
    "xgbclassifier__colsample_bylevel":[0.4,0.5,0.6],
    "xgbclassifier__reglambda":[0.4,0.4,0.6]


}

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)

grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

grid_search.fit(X_train,y_train)

best_model = grid_search.best_estimator_

y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)


print(classification_report(y_train,y_pred_train))
print(classification_report(y_test,y_pred_test))

joblib.dump(best_model, "best_machine_failure_model_v1.joblib")

repo_id = "Amidho/Machine-Failure-Prediction"
repo_type = "model"

api =HfApi(token=os.getenv("HF_TOKEN"))

try:
  api.repo_info(repo_id=repo_id,repo_type=repo_type)
  print(f"Space '{repo_id}' already exists")

except RepositoryNotFoundError:
  print(f"Space '{repo_id}' not found. Creating Space...")
  create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
  print(f"Space '{repo_id}' created.")


api.upload_file(
    path_or_fileobj="best_machine_failure_model_v1.joblib",
    path_in_repo="best_machine_failure_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type
)
