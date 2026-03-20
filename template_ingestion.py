import os
from pathlib import Path

project_name = "rag_src"

file_folder = [
    f"{project_name}/__init__.py",
    f"{project_name}/cofig/__init__.py",
    f"{project_name}/cofig/setting.py",
    f"{project_name}/ingestion/__init__.py",
    f"{project_name}/ingestion/loader.py",
    f"{project_name}/ingestion/splitter.py",
    f"{project_name}/ingestion/embedder.py",
    f"{project_name}/ingestion/pipeline.py",
    f"{project_name}/observability/__init__.py",
    f"{project_name}/observability/mlflow_tracker.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/exceptions.py",
    f"{project_name}/utils/logger.py",
    "main.py",
    "requirements.txt"
]

for file_path in file_folder:
    file_path = Path(file_path)

    file_dir, file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
    if (not os.path.exists(file_path) or (os.path.getsize(file_path)==0)):
        with open(file_path, 'w') as f:
            pass
    else:
        print(f"file is already at: {file_path}")


