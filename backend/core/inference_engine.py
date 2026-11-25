from contextlib import contextmanager, nullcontext
import time
import torch
from transformers.cache_utils import DynamicCache

from kv_cache.metrics import get_metrics_memory , get_compress_metrics , get_reconstruction_metrics
from kv_cache.utils import extract_keys_and_values

class InferenceEngine:
    def __init__(self , model_manager ,optimization_engine):
        self.model_manager = model_manager
        self.optimization_engine = optimization_engine

    @contextmanager
    def _xkv_sr_attention(self, model, optimizer, raw_kv_cache, metrics):
        from transformers.models.qwen3.modeling_qwen3 import (
            ALL_ATTENTION_FUNCTIONS,
            Qwen3Attention,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )

        model_body = getattr(model, "model", None)
        layers = getattr(model_body, "layers", None)
        if layers is None:
            raise TypeError("XKV-SR selective decoding currently requires Qwen3.")
        if len(layers) != optimizer.num_layers:
            raise ValueError("The XKV-SR cache layer count does not match the model.")

        source_keys, source_values = extract_keys_and_values(raw_kv_cache)
        original_forwards = []

        def make_sparse_forward(attention, layer_idx):
            def sparse_forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=None,
                **kwargs,
            ):
                if past_key_values is None:
                    raise ValueError("XKV-SR decoding requires a generation cache.")

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attention.head_dim)
                query_states = attention.q_norm(
                    attention.q_proj(hidden_states).view(hidden_shape)
                ).transpose(1, 2)
                key_states = attention.k_norm(
                    attention.k_proj(hidden_states).view(hidden_shape)
                ).transpose(1, 2)
                value_states = attention.v_proj(hidden_states).view(
                    hidden_shape
                ).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )
                generated_keys, generated_values = past_key_values.update(
                    key_states,
                    value_states,
                    attention.layer_idx,
                )

                reconstruction_start = time.perf_counter()
                prompt_keys, prompt_values, position_ids = optimizer.decode(
                    query_states, layer_idx
                )
                metrics["reconstruction_time"] += (
                    time.perf_counter() - reconstruction_start
                )

                original_key = source_keys[layer_idx]
                source_positions = position_ids.to(device=original_key.device)
                gather_index = source_positions.unsqueeze(-1).expand(
                    -1, -1, -1, original_key.shape[-1]
                )
                selected_original = torch.gather(
                    original_key, dim=-2, index=gather_index
                ).float()
                selected_reconstructed = prompt_keys.to(
                    device=original_key.device, dtype=torch.float32
                )
                key_error = (selected_original - selected_reconstructed).square()
                key_error_sum = key_error.sum()
                metrics["key_error_sum"] = (
                    key_error_sum
                    if metrics["key_error_sum"] is None
                    else metrics["key_error_sum"] + key_error_sum
                )
                key_reference_sum = selected_original.square().sum()
                metrics["key_reference_sum"] = (
                    key_reference_sum
                    if metrics["key_reference_sum"] is None
                    else metrics["key_reference_sum"] + key_reference_sum
                )
                metrics["key_element_count"] += key_error.numel()

                original_value = source_values[layer_idx]
                selected_original_value = torch.gather(
                    original_value,
                    dim=-2,
                    index=source_positions.unsqueeze(-1).expand(
                        -1, -1, -1, original_value.shape[-1]
                    ),
                ).float()
                selected_reconstructed_value = prompt_values.to(
                    device=original_value.device, dtype=torch.float32
                )
                value_error = (
                    selected_original_value - selected_reconstructed_value
                ).square()
                value_error_sum = value_error.sum()
                metrics["value_error_sum"] = (
                    value_error_sum
                    if metrics["value_error_sum"] is None
                    else metrics["value_error_sum"] + value_error_sum
                )
                value_reference_sum = selected_original_value.square().sum()
                metrics["value_reference_sum"] = (
                    value_reference_sum
                    if metrics["value_reference_sum"] is None
                    else metrics["value_reference_sum"] + value_reference_sum
                )
                metrics["value_element_count"] += value_error.numel()
                metrics["selected_token_fraction_sum"] += (
                    position_ids.shape[-1] / original_key.shape[-2]
                )
                metrics["decode_count"] += 1

                key_states = torch.cat(
                    (
                        prompt_keys.to(
                            device=generated_keys.device,
                            dtype=generated_keys.dtype,
                        ),
                        generated_keys,
                    ),
                    dim=-2,
                )
                value_states = torch.cat(
                    (
                        prompt_values.to(
                            device=generated_values.device,
                            dtype=generated_values.dtype,
                        ),
                        generated_values,
                    ),
                    dim=-2,
                )

                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                    attention.config._attn_implementation,
                    eager_attention_forward,
                )
                attn_output, attn_weights = attention_interface(
                    attention,
                    query_states,
                    key_states,
                    value_states,
                    None,
                    dropout=(
                        0.0
                        if not attention.training
                        else attention.attention_dropout
                    ),
                    scaling=attention.scaling,
                    sliding_window=attention.sliding_window,
                    **kwargs,
                )
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                return attention.o_proj(attn_output), attn_weights

            return sparse_forward

        try:
            for layer_idx, layer in enumerate(layers):
                attention = getattr(layer, "self_attn", None)
                if not isinstance(attention, Qwen3Attention):
                    raise TypeError(
                        "XKV-SR selective decoding requires Qwen3Attention layers."
                    )
                original_forwards.append((attention, attention.forward))
                attention.forward = make_sparse_forward(attention, layer_idx)
            yield
        finally:
            for attention, original_forward in original_forwards:
                attention.forward = original_forward

    @staticmethod
    def _xkv_sr_reconstruction_metrics(metrics):
        if not metrics["decode_count"]:
            return None

        key_mse = (
            metrics["key_error_sum"].item() / metrics["key_element_count"]
        )
        value_mse = (
            metrics["value_error_sum"].item() / metrics["value_element_count"]
        )
        return {
            "key_mse": key_mse,
            "value_mse": value_mse,
            "key_rmse": key_mse ** 0.5,
            "value_rmse": value_mse ** 0.5,
            "key_relative_error": (
                metrics["key_error_sum"].sqrt()
                / metrics["key_reference_sum"].sqrt().clamp_min(1e-12)
            ).item(),
            "value_relative_error": (
                metrics["value_error_sum"].sqrt()
                / metrics["value_reference_sum"].sqrt().clamp_min(1e-12)
            ).item(),
            "reconstruction_time": metrics["reconstruction_time"],
            "mse_scope": "query_selected_prompt_tokens",
            "selected_prompt_token_fraction": (
                metrics["selected_token_fraction_sum"]
                / metrics["decode_count"]
            ),
            "selective_decode_calls": metrics["decode_count"],
        }

    @staticmethod
    def _token_agreement_metrics(optimized_tokens, reference_tokens):
        compared_tokens = max(len(optimized_tokens), len(reference_tokens))
        matching_tokens = sum(
            optimized == reference
            for optimized, reference in zip(optimized_tokens, reference_tokens)
        )
        return {
            "token_agreement": (
                matching_tokens / compared_tokens if compared_tokens else None
            ),
            "matching_tokens": matching_tokens,
            "exact_sequence_match": optimized_tokens == reference_tokens,
            "compared_tokens": compared_tokens,
        }

    def generate(
        self,
        prompt: str,
        algorithm: str = "xkv",
        max_new_tokens: int = 100
    ):
        # 1. Lấy model và tokenizer
        model = self.model_manager.current_model
        tokenizer = self.model_manager.current_tokenizer
        # 2. Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        input_device = model.get_input_embeddings().weight.device

        inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
        }
        # 3. Prefill
        print("[1] Running Prefill Stage...")
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)

        raw_kv_cache = outputs.past_key_values
        # 4. Optimization
        print("[2] Running Optimizating ...")
        start_time_compress = time.perf_counter()
        self.optimization_engine.optimize(raw_kv_cache,algorithm)
        end_time_compress = time.perf_counter()
        optimizer = self.optimization_engine.get_optimizer(algorithm)
        memory_metrics = get_metrics_memory(raw_kv_cache,optimizer)
        from optimizers.xkv_sr.xkv_sr_optimizer import XKV_SROptimizer

        use_xkv_sr = isinstance(optimizer, XKV_SROptimizer)
        selective_metrics = {
            "key_error_sum": None,
            "key_reference_sum": None,
            "key_element_count": 0,
            "value_error_sum": None,
            "value_reference_sum": None,
            "value_element_count": 0,
            "selected_token_fraction_sum": 0.0,
            "decode_count": 0,
            "reconstruction_time": 0.0,
        }
        if use_xkv_sr:
            print("[3] Running XKV-SR selective decoding ...")
            optimized_past_kv = DynamicCache(config=model.config)
            reconstruction_metrics = None
            attention_mask = {
                layer_type: None for layer_type in set(model.config.layer_types)
            }
            attention_context = self._xkv_sr_attention(
                model, optimizer, raw_kv_cache, selective_metrics
            )
        else:
            print("[3] Running Decoding ...")
            start_time_reconstruction = time.perf_counter()
            optimized_past_kv = self.optimization_engine.reconstruct(model,algorithm)
            end_time_reconstruction = time.perf_counter()
            reconstruction_metrics = get_reconstruction_metrics(
                raw_kv_cache,
                optimized_past_kv,
                start_time_reconstruction,
                end_time_reconstruction,
            )
            reconstruction_metrics["mse_scope"] = "dense_full_cache"
            attention_context = nullcontext()
        # -------------------------------------------------------------
        # STAGE 4: AUTOREGRESSIVE DECODING
        # -------------------------------------------------------------
        print("[4] Running Autoregressive Decoding...")

        # The first token is identical for the optimized and full-cache paths.
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        optimized_token_sequence = [int(next_token_id.item())]

        past_key_values = optimized_past_kv
        generated_tokens = []

        # Nếu token đầu tiên đã là EOS thì dừng luôn
        if next_token_id.item() != tokenizer.eos_token_id:
            generated_tokens.append(next_token_id)
            current_input_ids = next_token_id

            with attention_context, torch.no_grad():
                for step in range(max_new_tokens - 1):
                    model_inputs = {
                        "input_ids": current_input_ids,
                        "past_key_values": past_key_values,
                        "use_cache": True,
                    }
                    if use_xkv_sr:
                        model_inputs["attention_mask"] = attention_mask
                        model_inputs["position_ids"] = torch.full(
                            current_input_ids.shape,
                            inputs["input_ids"].shape[-1] + step,
                            device=current_input_ids.device,
                            dtype=torch.long,
                        )
                    outputs = model(**model_inputs)
                    past_key_values = outputs.past_key_values

                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    optimized_token_sequence.append(int(next_token_id.item()))

                    # Kiểm tra EOS trước khi lưu
                    if next_token_id.item() == tokenizer.eos_token_id:
                        print(f"-> EOS detected at step {step + 2}")
                        break

                    generated_tokens.append(next_token_id)
                    current_input_ids = next_token_id
        else:
            print("-> EOS detected at step 1")

        if use_xkv_sr:
            reconstruction_metrics = self._xkv_sr_reconstruction_metrics(
                selective_metrics
            )

        reference_tokens = [optimized_token_sequence[0]]
        reference_past_key_values = raw_kv_cache
        reference_input_ids = torch.tensor(
            [[reference_tokens[0]]], device=input_device, dtype=torch.long
        )
        reference_start_time = time.perf_counter()
        if reference_tokens[0] != tokenizer.eos_token_id:
            with torch.no_grad():
                for _ in range(max_new_tokens - 1):
                    reference_outputs = model(
                        input_ids=reference_input_ids,
                        past_key_values=reference_past_key_values,
                        use_cache=True,
                    )
                    reference_past_key_values = reference_outputs.past_key_values
                    reference_next_token = torch.argmax(
                        reference_outputs.logits[:, -1, :],
                        dim=-1,
                        keepdim=True,
                    )
                    reference_tokens.append(int(reference_next_token.item()))
                    if reference_next_token.item() == tokenizer.eos_token_id:
                        break
                    reference_input_ids = reference_next_token
        accuracy_metrics = self._token_agreement_metrics(
            optimized_token_sequence, reference_tokens
        )
        accuracy_metrics["reference_generation_time"] = (
            time.perf_counter() - reference_start_time
        )

        # Ghép token (xử lý trường hợp không sinh được token nào)
        if generated_tokens:
            generated_ids = torch.cat(generated_tokens, dim=-1)
        else:
            generated_ids = torch.empty((1, 0), dtype=torch.long, device=input_device)

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # 6. Collect metrics
        compress_metrics = get_compress_metrics(memory_metrics["memory_raw_cache"],memory_metrics["memory_compressed_cache"],
                                                start_time_compress,end_time_compress,self.optimization_engine.get_optimizer(algorithm))
        result = {
            "memory_metrics" : memory_metrics,
            "compress_metrics" : compress_metrics,
            "reconstruction_metrics" : reconstruction_metrics,
            "accuracy_metrics": accuracy_metrics,
            "Generated_tokens" : generated_ids.shape[-1],
            "generated_text" : generated_text,
            "reference_generated_text": tokenizer.decode(
                torch.tensor(reference_tokens, device=input_device),
                skip_special_tokens=True,
            ),
        }
        # 7. Return result
        return result