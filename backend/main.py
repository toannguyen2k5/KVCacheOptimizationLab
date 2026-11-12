# Place in: inference_engine.py
import torch
from kv_cache.utils import get_kv_cache_memory_mb
from optimizers.xkv.xkv_optimizer import XKVOptimizer
from models.qwen3_loader import load_qwen3_model_and_tokenizer
import time
import numpy as np

def run_xkv_demo(prompt: str):
    model, tokenizer = load_qwen3_model_and_tokenizer()
    inputs = tokenizer(prompt, return_tensors="pt")
    
    optimizer = XKVOptimizer()

    # -------------------------------------------------------------
    # STAGE 1: PREFILL (Xử lý toàn bộ Prompt đầu vào)
    # -------------------------------------------------------------
    print("[1] Running Prefill Stage...")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    
    raw_kv_cache = outputs.past_key_values
    raw_mem = get_kv_cache_memory_mb(raw_kv_cache)
    print(f"-> Prefill completed. Original KV Cache Memory: {raw_mem:.2f} MB")

    # STAGE 2
    optimizer = XKVOptimizer(window_size=4, grouping="window")   # thử thêm grouping="cka"
    optimizer.compress(raw_kv_cache)
    comp_mem_mb = optimizer.memory_mb()
    print(f"-> Compressed: {comp_mem_mb:.2f} MB (Saved {(raw_mem - comp_mem_mb)/raw_mem*100:.1f}%)")
    print("Nhóm layer:", [g["ids"] for g in optimizer.groups])

    # STAGE 3: dựng lại cache (tạm thời, để kiểm chứng chất lượng)
    from transformers.cache_utils import DynamicCache
    optimized_past_kv = DynamicCache()
    for i in range(optimizer.num_layers):
        k, v = optimizer.reconstruct_layer(i)
        optimized_past_kv.update(k.to(model.dtype), v.to(model.dtype), layer_idx=i)

if __name__ == "__main__":
    base_text = "Hãy giải thích khái niệm KV Cache trong mô hình ngôn ngữ lớn một cách rõ ràng và chi tiết, đồng thời đưa ra ví dụ minh họa."
    base_text *= 5
    run_xkv_demo(base_text)