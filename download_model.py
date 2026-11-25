from huggingface_hub import snapshot_download
from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent
# Nạp file .env
load_dotenv(BASE_DIR / ".env")

model_id = "Qwen/Qwen3-8B"
local_dir = os.getenv("MODEL_PATH")

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
)

print(f"Model downloaded to: {local_dir}")
