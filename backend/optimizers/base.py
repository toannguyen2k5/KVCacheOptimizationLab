from abc import ABC, abstractmethod

class BaseOptimizer(ABC):

    @abstractmethod
    def compress(self, past_key_values):
        pass

    @abstractmethod
    def reconstruct_layer(self, layer_idx):
        pass

    @abstractmethod
    def reconstruct_cache(self, model):
        pass

    @abstractmethod
    def memory_mb(self):
        pass