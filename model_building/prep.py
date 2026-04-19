import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import login, HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/Amidho/Machine-Failure-Prediction/machine-failure-prediction.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset Loaded Sucess!")

df.drop(columns = ["UDI"], inplace=True)

label_encoder = LabelEncoder()
df['Type'] = label_encoder.fit_transform(df['Type'])

target = 'Failure'

X = df.drop(columns=[target],inplace=True)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train.to_csv("Xtrain.csv",index=False)
X_test.to_csv("Xtest.csv",index=False)
y_train.to_csv("ytrain.csv",index=False)
y_test.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
  api.upload_file(
      path_or_fileobj=file_path,
      path_in_repo=file_path.split("/")[-1],
      repo_id="Amidho/Machine-Failure-Prediction",
      repo_type="dataset"
  )
