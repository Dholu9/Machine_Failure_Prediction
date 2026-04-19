from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "Amidho/Machine-Failure-Prediction"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
  api.repo_info(repo_id=repo_id, repo_type=repo_type)
  print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
  print(f"Space '{repo_id}' not found. Creating new space...")
  create_repo(repo_id=repo_id, repo_type=repo_type,private=False)
  print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="machineprediction/data",
    repo_id=repo_id,
    repo_type=repo_type
)
