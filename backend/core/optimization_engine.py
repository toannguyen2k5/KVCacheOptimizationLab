class OptimizationEngine:

    def __init__(self):
        self.optimizers = {}

    def register(self, name, optimizer):
        self.optimizers[name] = optimizer

    def get_optimizer(self, name):
        if name not in self.optimizers:
            raise ValueError(
                f"Optimizer '{name}' chưa được đăng ký"
            )
        return self.optimizers[name]

    def get_all_optimizer(self):
        return self.optimizers.copy()

    def optimize(self, kv_cache, algorithm="xkv"):
        optimizer = self.get_optimizer(algorithm)
        # Thực hiện nén
        optimizer.compress(kv_cache)
        return optimizer

    def reconstruct(self, model, algorithm="xkv"):
        optimizer = self.get_optimizer(algorithm)

        return optimizer.reconstruct_cache(model)