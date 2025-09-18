import torch.nn as nn
from .base import Embedding


class LinearEmbedding(Embedding):
    def __init__(self, input_dim, embedding_dim):
        super().__init__(input_dim, embedding_dim)
        
    def initialize_layers(self):  
        # Use LazyLinear to infer input features at first call
        self.layers = nn.LazyLinear(self.embedding_dim) 
