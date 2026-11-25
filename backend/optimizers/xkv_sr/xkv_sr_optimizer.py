"""Cross-layer KV compression with landmark-based selective reconstruction."""

import math
from collections.abc import Mapping

import torch
from transformers.cache_utils import DynamicCache

from kv_cache.utils import extract_keys_and_values
from ..xkv.xkv_optimizer import XKVOptimizer


class XKV_SROptimizer(XKVOptimizer):
    """xKV low-rank compression plus query-guided chunk selection.

    ``compress`` and ``reconstruct_cache`` preserve the existing optimizer
    interface. ``decode`` exposes the selective path to attention code that has
    the current query states and can use the returned original token positions.
    """

    def __init__(
        self,
        model=None,
        config=None,
        rank_ratio_k=0.09,
        rank_ratio_v=0.14,
        window_size=4,
        grouping="window",
        cka_threshold=0.85,
        sparse_budget=2048,
        chunk_size=8,
        local_window_size=None,
    ):
        options = config if isinstance(config, Mapping) else {}

        def option(name, value):
            return options.get(name, getattr(config, name, value))

        rank_ratio_k = option("rank_ratio_k", rank_ratio_k)
        rank_ratio_v = option("rank_ratio_v", rank_ratio_v)
        window_size = option("window_size", window_size)
        grouping = option("grouping", grouping)
        cka_threshold = option("cka_threshold", cka_threshold)
        sparse_budget = option("sparse_budget", sparse_budget)
        chunk_size = option("chunk_size", chunk_size)
        local_window_size = option("local_window_size", local_window_size)

        super().__init__(
            rank_ratio_k=rank_ratio_k,
            rank_ratio_v=rank_ratio_v,
            window_size=window_size,
            grouping=grouping,
            cka_threshold=cka_threshold,
        )

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if sparse_budget < 0:
            raise ValueError("sparse_budget cannot be negative")

        self.model = model
        self.chunk_size = int(chunk_size)
        self.sparse_budget = int(sparse_budget)
        self.local_window_size = (
            int(local_window_size)
            if local_window_size is not None
            else 4 * self.chunk_size
        )
        if self.local_window_size < 0:
            raise ValueError("local_window_size cannot be negative")

        self.k_landmark = None
        self.k_landmark_idx = None
        self.landmarks = {}
        self.landmark_indices = {}
        self._local_starts = {}
        self._batch_size = 0
        self._sequence_length = 0
        self._key_dtype = None
        self._value_dtype = None

    @staticmethod
    def _factorize(matrices, rank_ratio):
        """Factor [batch * sequence, heads * head_dim] matrices jointly."""
        feature_size = matrices[0].shape[1]
        combined = torch.cat([matrix.float() for matrix in matrices], dim=1)
        rank = min(
            max(1, int(combined.shape[1] * rank_ratio)),
            min(combined.shape),
        )
        u, singular_values, vh = torch.linalg.svd(combined, full_matrices=False)
        shared = (u[:, :rank] * singular_values[:rank]).to(matrices[0].dtype)
        right_factors = [
            vh[:rank, index * feature_size:(index + 1) * feature_size].to(
                matrices[index].dtype
            )
            for index in range(len(matrices))
        ]
        return shared, right_factors

    def compress(self, past_key_values):
        keys, values = extract_keys_and_values(past_key_values)
        if not keys:
            raise ValueError("Cannot compress an empty KV cache")
        if len(keys) != len(values):
            raise ValueError("Key and value caches must have the same number of layers")

        first_shape = tuple(keys[0].shape)
        if len(first_shape) != 4:
            raise ValueError("KV tensors must have shape [batch, heads, sequence, head_dim]")
        if first_shape[-2] == 0:
            raise ValueError("Cannot compress a KV cache with an empty sequence")

        for layer_idx, (key, value) in enumerate(zip(keys, values)):
            if key.ndim != 4 or value.ndim != 4:
                raise ValueError(
                    f"Layer {layer_idx} KV tensors must have four dimensions"
                )
            if tuple(key.shape) != first_shape or tuple(value.shape) != first_shape:
                raise ValueError(
                    "All layers must have matching key/value shapes for cross-layer SVD"
                )
            if key.device != keys[0].device or value.device != values[0].device:
                raise ValueError("All KV layers must be on the same device")

        batch_size, self.H, self._sequence_length, self.D = first_shape
        self._batch_size = batch_size
        self._key_dtype = keys[0].dtype
        self._value_dtype = values[0].dtype
        self.num_layers = len(keys)

        matrices_k = [
            key.permute(0, 2, 1, 3).reshape(
                batch_size * self._sequence_length, self.H * self.D
            )
            for key in keys
        ]
        matrices_v = [
            value.permute(0, 2, 1, 3).reshape(
                batch_size * self._sequence_length, self.H * self.D
            )
            for value in values
        ]

        self.groups = []
        self.layer_to_group = {}
        for group_idx, layer_ids in enumerate(self._make_groups(keys, values)):
            shared_k, factors_k = self._factorize(
                [matrices_k[index] for index in layer_ids], self.rank_ratio_k
            )
            shared_v, factors_v = self._factorize(
                [matrices_v[index] for index in layer_ids], self.rank_ratio_v
            )
            self.groups.append(
                {
                    "ids": layer_ids,
                    "Ak": shared_k,
                    "Bk": factors_k,
                    "Av": shared_v,
                    "Bv": factors_v,
                }
            )
            for group_position, layer_idx in enumerate(layer_ids):
                self.layer_to_group[layer_idx] = (group_idx, group_position)

        self.build_landmarks(keys)
        self.k_landmark = [self.landmarks[index] for index in range(self.num_layers)]
        self.k_landmark_idx = [
            self.landmark_indices[index] for index in range(self.num_layers)
        ]
        return self.groups

    def build_landmarks(self, key_cache):
        """Mean-pool full key chunks, retaining a recent local window verbatim."""
        keys = [key_cache] if isinstance(key_cache, torch.Tensor) else list(key_cache)
        self.landmarks = {}
        self.landmark_indices = {}
        self._local_starts = {}

        for layer_idx, key in enumerate(keys):
            if key.ndim != 4:
                raise ValueError(
                    "Each key tensor must have shape [batch, heads, sequence, head_dim]"
                )
            batch_size, num_heads, sequence_length, head_dim = key.shape
            selectable_chunks = max(
                0,
                (sequence_length - self.local_window_size) // self.chunk_size,
            )
            local_start = selectable_chunks * self.chunk_size

            if selectable_chunks:
                chunk_keys = key[:, :, :local_start].reshape(
                    batch_size,
                    num_heads,
                    selectable_chunks,
                    self.chunk_size,
                    head_dim,
                )
                landmark = chunk_keys.float().mean(dim=-2).to(key.dtype)
            else:
                landmark = key.new_empty((batch_size, num_heads, 0, head_dim))

            indices = torch.arange(
                selectable_chunks, device=key.device, dtype=torch.long
            ).view(1, 1, -1).expand(batch_size, num_heads, -1)
            self.landmarks[layer_idx] = landmark
            self.landmark_indices[layer_idx] = indices
            self._local_starts[layer_idx] = local_start

        return self.landmarks

    def select_chunks(self, query_states, layer_idx):
        """Return per-batch, per-KV-head chunk IDs selected by query relevance.

        ``query_states`` must have shape [batch, query_heads, query_length, head_dim].
        Selection scores are softmax-normalized over chunks, summed over query
        positions, and max-reduced across query heads sharing a KV head.
        """
        if layer_idx not in self.landmarks:
            raise ValueError(f"No landmarks are available for layer {layer_idx}")
        if query_states.ndim != 4:
            raise ValueError(
                "query_states must have shape [batch, heads, query, head_dim]"
            )

        landmarks = self.landmarks[layer_idx]
        batch_size, kv_heads, num_chunks, head_dim = landmarks.shape
        if query_states.shape[0] != batch_size or query_states.shape[-1] != head_dim:
            raise ValueError("Query and key landmark batch/head dimensions must match")
        if query_states.shape[1] % kv_heads != 0:
            raise ValueError("Query head count must be divisible by KV head count")

        query_groups = query_states.shape[1] // kv_heads
        selected_count = min(
            num_chunks,
            self.sparse_budget // self.chunk_size,
        )
        if selected_count == 0:
            return torch.empty(
                (batch_size, kv_heads, 0),
                device=query_states.device,
                dtype=torch.long,
            )

        queries = query_states.reshape(
            batch_size,
            kv_heads,
            query_groups,
            query_states.shape[-2],
            head_dim,
        ).float()
        scores = torch.einsum(
            "bhgqd,bhcd->bhgqc",
            queries,
            landmarks.to(device=query_states.device, dtype=torch.float32),
        ) / math.sqrt(head_dim)
        probabilities = torch.softmax(scores, dim=-1)
        relevance = probabilities.sum(dim=-2).amax(dim=-2)
        selected = relevance.topk(selected_count, dim=-1).indices
        return selected.sort(dim=-1).values

    def _reconstruct_positions(self, layer_idx, position_ids):
        group_idx, group_position = self.layer_to_group[layer_idx]
        group = self.groups[group_idx]

        def reconstruct(shared, factors):
            right = factors[group_position].reshape(
                shared.shape[-1], self.H, self.D
            ).permute(1, 0, 2)
            shared_by_batch = shared.reshape(
                self._batch_size, self._sequence_length, shared.shape[-1]
            )
            gather_indices = position_ids.unsqueeze(-1).expand(
                -1, -1, -1, shared.shape[-1]
            )
            selected_shared = torch.gather(
                shared_by_batch.unsqueeze(1).expand(
                    -1, self.H, -1, -1
                ),
                dim=2,
                index=gather_indices,
            )
            return torch.einsum("bhtr,hrd->bhtd", selected_shared, right)

        return reconstruct(group["Ak"], group["Bk"]), reconstruct(
            group["Av"], group["Bv"]
        )

    def reconstruct_layer(self, layer_idx):
        if layer_idx not in self.layer_to_group:
            raise ValueError(f"Layer {layer_idx} has not been compressed")

        group_idx, group_position = self.layer_to_group[layer_idx]
        group = self.groups[group_idx]

        def reconstruct(shared, factors):
            matrix = shared.float() @ factors[group_position].float()
            return matrix.reshape(
                self._batch_size,
                self._sequence_length,
                self.H,
                self.D,
            ).permute(0, 2, 1, 3)

        return reconstruct(group["Ak"], group["Bk"]), reconstruct(
            group["Av"], group["Bv"]
        )

    def decode(self, query_states, layer_idx):
        """Reconstruct only retrieved chunks plus the recent local window.

        Returns ``(keys, values, position_ids)``. The original token positions
        are provided so an attention implementation can preserve positional and
        causal-mask semantics when consuming this sparse cache.
        """
        chunk_ids = self.select_chunks(query_states, layer_idx)
        chunk_positions = (
            chunk_ids.unsqueeze(-1) * self.chunk_size
            + torch.arange(
                self.chunk_size,
                device=query_states.device,
                dtype=torch.long,
            ).view(1, 1, -1)
        ).flatten(start_dim=-2)

        local_start = self._local_starts[layer_idx]
        local_positions = torch.arange(
            local_start,
            self._sequence_length,
            device=query_states.device,
            dtype=torch.long,
        ).view(1, 1, -1).expand(
            self._batch_size, self.H, -1
        )
        position_ids = torch.cat((chunk_positions, local_positions), dim=-1)
        keys, values = self._reconstruct_positions(layer_idx, position_ids)
        return keys, values, position_ids

    def reconstruct_cache(self, model=None):
        optimized_cache = DynamicCache()
        model_dtype = getattr(model, "dtype", None)
        for layer_idx in range(self.num_layers):
            key, value = self.reconstruct_layer(layer_idx)
            key_dtype = model_dtype if isinstance(model_dtype, torch.dtype) else self._key_dtype
            value_dtype = model_dtype if isinstance(model_dtype, torch.dtype) else self._value_dtype
            optimized_cache.update(
                key.to(dtype=key_dtype),
                value.to(dtype=value_dtype),
                layer_idx=layer_idx,
            )
        return optimized_cache

    def memory_mb(self):
        factor_bytes = super().memory_mb() * (1024 ** 2)
        metadata_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in list(self.landmarks.values())
            + list(self.landmark_indices.values())
        )
        return (factor_bytes + metadata_bytes) / (1024 ** 2)
