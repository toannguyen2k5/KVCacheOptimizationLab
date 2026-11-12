from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen3-8B"
local_dir = r"D:\KVCacheOptimizationLab\weights"

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
)

print(f"Model downloaded to: {local_dir}")
