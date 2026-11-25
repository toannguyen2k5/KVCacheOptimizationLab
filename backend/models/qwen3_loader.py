import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from dotenv import load_dotenv
import json

BASE_DIR = Path(__file__).resolve().parents[2]
# Nạp file .env
load_dotenv(BASE_DIR / ".env")
# Đường dẫn mặc định trỏ đến thư mục chứa trọng số mô hình
class Model_Qwen:
    def __init__(self):
        self.model_paths = json.loads(
            os.getenv("MODEL_PATH", "{}")
        )
        self.current_name_version = None
        self.current_model = None
        self.current_tokenizer = None

    def load_model(self,model_version : str = "qwen3-8b"):
        """
        Load mô hình Qwen3-8B và Tokenizer từ thư mục trọng số cục bộ chạy trên CPU.
        """
        # kiểm tra version hiện tại
        if self.current_name_version == model_version:
            return self.current_model , self.current_tokenizer
        if model_version not in self.model_paths:
            raise ValueError(
                f"Không tìm thấy phiên bản {model_version}"
            )
        # xóa model hiện tại và xóa cache nếu có
        del self.current_model
        torch.cuda.empty_cache()

        model_path = self.model_paths[model_version]

        print(f"Loading model qwen version {model_version} ở đường dẫn: {model_path}")
        self.current_tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )

        print(f"[Loading] Loading Qwen3 Model onto CPU (float16)...")
        # Sử dụng torch.float16 để tiết kiệm 50% RAM CPU so với float32
        # low_cpu_mem_usage=True giúp giảm RAM ảo khi đang load safetensors
        self.current_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=os.getenv("DEVICE"),
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        self.current_model.eval()  # Đưa model về chế độ evaluation (tắt dropout/batchnorm)
        print("[Success] Model and Tokenizer loaded successfully!")
        
        return self.current_model, self.current_tokenizer