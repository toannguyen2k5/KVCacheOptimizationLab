from models.qwen3_loader import Model_Qwen
from core.optimization_engine import OptimizationEngine
from core.inference_engine import InferenceEngine
from optimizers.xkv.xkv_optimizer import XKVOptimizer
from optimizers.xkv_sr.xkv_sr_optimizer import XKV_SROptimizer
import torch

def main(promt:str):
    model_manager = Model_Qwen()
    optimization_engine = OptimizationEngine()
    XKV_Optimizer = XKVOptimizer(rank_ratio_k =0.09 , rank_ratio_v = 0.14 ,window_size = 4 , grouping = "window", cka_threshold = 0.6 )
    XKVSROptimizer = XKV_SROptimizer(rank_ratio_k =0.09 , rank_ratio_v = 0.14 ,window_size = 4 , grouping = "window", cka_threshold = 0.6)
    optimization_engine.register("xkv",XKV_Optimizer)
    optimization_engine.register("xkv_sr",XKVSROptimizer)
    engine = InferenceEngine(
        model_manager=model_manager,
        optimization_engine=optimization_engine
    )

    model_manager.load_model() # mặc định model qwen3 8b

    result = engine.generate(
        prompt= promt,
        algorithm="xkv",
        max_new_tokens=200
    )

    print(result)
    
if __name__ == "__main__":
    prompt = """
    Artificial Intelligence has become one of the most transformative
    technologies of the modern era, influencing nearly every aspect
    of human life, from education and healthcare to transportation,
    communication, scientific research, and business operations.

    Most modern LLMs are built upon the Transformer architecture,
    which relies heavily on the self-attention mechanism.

    During text generation, the model processes input tokens and
    computes Query, Key, and Value representations at each Transformer
    layer. To address redundant computations, modern Transformer-based
    models use a technique known as KV Cache.

    During the Prefill phase, the model processes the entire prompt
    and generates Key and Value tensors for every Transformer layer.
    These tensors are stored in the KV Cache.

    During the Decode phase, the model generates one token at a time.
    The newly generated Key and Value tensors are appended to the
    existing cache, allowing the model to reuse previously computed
    information.

    Although KV Cache improves inference efficiency, its memory
    consumption increases with sequence length, number of layers,
    attention heads, and head dimension.

    Researchers have proposed several optimization techniques,
    including token eviction, quantization, low-rank approximation,
    Singular Value Decomposition (SVD), and layer similarity analysis
    using Centered Kernel Alignment (CKA).

    An effective KV Cache optimization method should reduce memory
    consumption while preserving reconstruction quality and
    maintaining efficient autoregressive decoding.

    Explain these concepts in detail, including the Transformer
    architecture, self-attention, Query, Key, Value, Prefill,
    Decode, KV Cache, CKA, SVD, and low-rank approximation.
    Also discuss how to evaluate compression ratio, memory reduction,
    reconstruction error, reconstruction time, inference latency,
    and tokens per second.
"""

    main(prompt)