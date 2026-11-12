import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Đường dẫn mặc định trỏ đến thư mục chứa trọng số mô hình
DEFAULT_MODEL_PATH = "D:\\KVCacheOptimizationLab\\weights\\Qwen3-8B"

def load_qwen3_model_and_tokenizer(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load mô hình Qwen3-8B và Tokenizer từ thư mục trọng số cục bộ chạy trên CPU.
    """
    print("path model = ", model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy thư mục trọng số tại: {model_path}")

    print(f"[Loading] Loading Tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )

    print(f"[Loading] Loading Qwen3 Model onto CPU (float16)...")
    # Sử dụng torch.float16 để tiết kiệm 50% RAM CPU so với float32
    # low_cpu_mem_usage=True giúp giảm RAM ảo khi đang load safetensors
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    model.eval()  # Đưa model về chế độ evaluation (tắt dropout/batchnorm)
    print("[Success] Model and Tokenizer loaded successfully!")
    
    return model, tokenizer

if __name__ == "__main__":
    # Test chạy thử độc lập hàm load
    model, tokenizer = load_qwen3_model_and_tokenizer()
    print(f"Model Architecture: {model.__class__.__name__}")