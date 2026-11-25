import torch
from kv_cache.utils import extract_keys_and_values


def analyze_tensor(tensor):
    """
    Phân tích thông tin cơ bản của một tensor.
    """
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "num_elements": tensor.numel(),
        "element_size": tensor.element_size(),
        "memory_bytes": tensor.numel() * tensor.element_size()
    }

def analyze_layer(key, value, layer_idx):
    """
    Phân tích K và V của một layer.
    """
    return {
        "layer_idx": layer_idx,
        "key": analyze_tensor(key),
        "value": analyze_tensor(value),
        "sequence_length": key.shape[-2],
        "num_key_value_heads": key.shape[1],
        "head_dim": key.shape[-1]
    }

def get_cache_shapes(past_key_values):
    """
    Lấy shape K và V của tất cả layer.
    """
    keys, values = extract_keys_and_values(past_key_values)
    return [
        {
            "layer_idx": i,
            "key_shape": tuple(k.shape),
            "value_shape": tuple(v.shape)
        }
        for i, (k, v) in enumerate(zip(keys, values))
    ]

def analyze_kv_cache(past_key_values):
    """
    Phân tích toàn bộ KV Cache.
    """
    keys, values = extract_keys_and_values(past_key_values)
    layers = [
        analyze_layer(k, v, i)
        for i, (k, v) in enumerate(zip(keys, values))
    ]
    return {
        "num_layers": len(layers),
        "layers": layers
    }

def get_cache_statistics(past_key_values):
    """
    Thống kê các đặc điểm chung của KV Cache.
    """
    keys, values = extract_keys_and_values(past_key_values)
    if not keys:
        return {}
    return {
        "num_layers": len(keys),
        "sequence_length": keys[0].shape[-2],
        "num_key_value_heads": keys[0].shape[1],
        "head_dim": keys[0].shape[-1],
        "key_dtype": str(keys[0].dtype),
        "value_dtype": str(values[0].dtype)
    }

def print_cache_summary(past_key_values):
    """
    In thông tin tổng quan KV Cache.
    """
    stats = get_cache_statistics(past_key_values)
    print("\n========== KV CACHE SUMMARY ==========")
    for name, value in stats.items():
        print(f"{name}: {value}")
    print("======================================\n")