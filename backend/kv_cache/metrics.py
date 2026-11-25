import torch
from kv_cache.utils import extract_keys_and_values

""" --------------------------------------    các hàm lấy chỉ số bộ nhớ      ------------------------------"""
def get_kv_cache_memory(past_key_values):
    """
    Tính tổng dung lượng K và V của toàn bộ KV Cache.
    """
    keys, values = extract_keys_and_values(past_key_values)
    total_bytes = 0
    for key, value in zip(keys, values):
        total_bytes += key.nelement() * key.element_size()
        total_bytes += value.nelement() * value.element_size()
    return total_bytes / (1024 ** 2)

def get_metrics_memory (past_key_values,optimizer):
    # Original KV Size (dung dượng kv cache ban đầu)
    m_raw = get_kv_cache_memory(past_key_values)

    # Compressed KV size (dung dượng kv cache lúc nén)
    m_compressed = optimizer.memory_mb()

    # Memory saved (dung lượng tiết kiệm được)
    m_saved = m_raw - m_compressed

    # memory reduction (phần trăm bộ nhớ giảm)
    m_reduction = m_saved / m_raw

    # peak memory (bộ nhớ mà gpu xử dụng tối đa trong quá trình chạy)
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "memory_raw_cache" : m_raw,
        "memory_compressed_cache" : m_compressed,
        "memory_saved_cache" : m_saved,
        "memory_reduction_cache" : m_reduction,
        "peak_memory" : peak_memory
    }
"""----------------------------------- các hàm lấy chỉ số nén -----------------------------------------"""
def get_compress_metrics(m_raw,m_compressed,time_start,time_end,optimizer):
    return {
        "compression_raito" : m_raw / m_compressed,
        "compression_time" : time_end - time_start,
        "number_of_groups" : len(optimizer.groups),
        "rank_raito_key" :  optimizer.rank_ratio_k,
        "rank_raito_value": optimizer.rank_ratio_v,
    }
"""---------------------------------- các hàm lấy chỉ số khôi phục -----------------------------------"""

import torch


def calculate_cka(original, reconstructed, eps=1e-12):
    """
    CKA theo từng batch và attention head.

    Input:
        original:      [B, H, S, D]
        reconstructed: [B, H, S, D]

    Output:
        cka_per_head: [B, H]
        cka_mean: float
    """

    assert original.shape == reconstructed.shape

    B, H, S, D = original.shape

    X = original.float()
    Y = reconstructed.float()

    # Centering matrix
    I = torch.eye(S, device=X.device)
    ones = torch.ones(S, S, device=X.device)
    H_center = I - ones / S

    cka_per_head = torch.zeros(
        B, H,
        device=X.device,
        dtype=torch.float32
    )

    for b in range(B):
        for h in range(H):

            x = X[b, h]  # [S, D]
            y = Y[b, h]  # [S, D]

            # Gram matrices
            K = x @ x.T
            L = y @ y.T

            # Centering
            Kc = H_center @ K @ H_center
            Lc = H_center @ L @ H_center

            # CKA
            numerator = torch.sum(Kc * Lc)

            denominator = (
                torch.norm(Kc, p="fro")
                * torch.norm(Lc, p="fro")
            )

            cka_per_head[b, h] = (
                numerator / denominator.clamp_min(eps)
            )

    return cka_per_head, cka_per_head.mean().item()


def calculate_reconstruction_cka(
    original_keys,
    reconstructed_keys,
    original_values,
    reconstructed_values
):
    """
    Tính CKA riêng cho Key và Value trên toàn bộ layer.
    """

    assert len(original_keys) == len(reconstructed_keys)
    assert len(original_values) == len(reconstructed_values)

    key_scores = []
    value_scores = []

    for i in range(len(original_keys)):

        # Key CKA
        _, key_cka = calculate_cka(
            original_keys[i],
            reconstructed_keys[i]
        )

        # Value CKA
        _, value_cka = calculate_cka(
            original_values[i],
            reconstructed_values[i]
        )

        key_scores.append(key_cka)
        value_scores.append(value_cka)

    return {
        "key_cka_per_layer": key_scores,
        "value_cka_per_layer": value_scores,

        "key_cka_mean": sum(key_scores) / len(key_scores),
        "value_cka_mean": sum(value_scores) / len(value_scores),

        "overall_cka": (
            sum(key_scores) + sum(value_scores)
        ) / (len(key_scores) + len(value_scores))
    }


def get_reconstruction_metrics(
    original,
    reconstructed,
    time_start,
    time_end
):
    # Lấy Key và Value từ hai DynamicCache
    original_keys, original_values = extract_keys_and_values(original)

    reconstructed_keys, reconstructed_values = extract_keys_and_values(
        reconstructed
    )

    assert len(original_keys) == len(reconstructed_keys)
    assert len(original_values) == len(reconstructed_values)

    key_mse = []
    value_mse = []

    key_relative = []
    value_relative = []

    # Tính MSE và Relative Error cho từng layer
    for i in range(len(original_keys)):

        # ---------- KEY ----------
        k_original = original_keys[i].float()
        k_reconstructed = reconstructed_keys[i].float()

        k_error = k_original - k_reconstructed

        key_mse.append(
            torch.mean(k_error ** 2).item()
        )

        key_relative.append(
            (
                torch.norm(k_error) /
                torch.norm(k_original).clamp_min(1e-12)
            ).item()
        )

        # ---------- VALUE ----------
        v_original = original_values[i].float()
        v_reconstructed = reconstructed_values[i].float()

        v_error = v_original - v_reconstructed

        value_mse.append(
            torch.mean(v_error ** 2).item()
        )

        value_relative.append(
            (
                torch.norm(v_error) /
                torch.norm(v_original).clamp_min(1e-12)
            ).item()
        )

    # Trung bình trên tất cả layer
    key_mse_mean = sum(key_mse) / len(key_mse)
    value_mse_mean = sum(value_mse) / len(value_mse)

    key_relative_mean = sum(key_relative) / len(key_relative)
    value_relative_mean = sum(value_relative) / len(value_relative)

    # CKA riêng Key và Value
    cka_results = calculate_reconstruction_cka(
        original_keys,
        reconstructed_keys,
        original_values,
        reconstructed_values
    )

    return {
        "key_mse": key_mse_mean,
        "value_mse": value_mse_mean,

        "key_rmse": key_mse_mean ** 0.5,
        "value_rmse": value_mse_mean ** 0.5,

        "key_relative_error": key_relative_mean,
        "value_relative_error": value_relative_mean,

        "cka": cka_results,

        "reconstruction_time": time_end - time_start
    }
