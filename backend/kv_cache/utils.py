def extract_keys_and_values(past_key_values):
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        keys = list(past_key_values.key_cache)
        values = list(past_key_values.value_cache)
    else:
        layers = getattr(past_key_values, "layers", None)
        if layers is None:
            try:
                layers = list(past_key_values)
            except TypeError as exc:
                raise TypeError(
                    "past_key_values must be a cache object or an iterable of key/value layers"
                ) from exc

        keys, values = [], []
        for index, item in enumerate(layers):
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                key, value = item[0], item[1]
            elif hasattr(item, "keys") and hasattr(item, "values"):
                key, value = item.keys, item.values
            elif hasattr(item, "key_cache") and hasattr(item, "value_cache"):
                key, value = item.key_cache, item.value_cache
            elif hasattr(item, "key") and hasattr(item, "value"):
                key, value = item.key, item.value
            else:
                raise TypeError(
                    f"Unsupported key/value cache layer at index {index}: "
                    f"{type(item).__name__}"
                )
            keys.append(key)
            values.append(value)

    if len(keys) != len(values):
        raise ValueError("The key and value caches must contain the same number of layers")
    return keys, values

def get_kv_cache_memory_mb(past_key_values):
    """Tính dung lượng bộ nhớ của past_key_values gốc."""

    keys, values = extract_keys_and_values(past_key_values)
    total_bytes = 0

    for k, v in zip(keys, values):
        total_bytes += k.nelement() * k.element_size()
        total_bytes += v.nelement() * v.element_size()

    return total_bytes / (1024 * 1024)