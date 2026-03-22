import os
from pathlib import Path

project_name = "rag_src"

file_folder = [
    f"{project_name}/caching/__init__.py",
    f"{project_name}/caching/cache_manager.py",
    f"{project_name}/evaluation/__init__.py",
    f"{project_name}/evaluation/ragas_evaluator.py",
    f"{project_name}/utils/retry.py"
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