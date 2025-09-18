import torch
import opt_einsum as oe
from einops import repeat, rearrange

from models.decompssm.functional.krylov import krylov
from models.decompssm.ssm.companion import CompanionSSM


class ShiftSSM(CompanionSSM):
    """
    Open-loop implementation of Shift SSM:
    -> y_t = C \sum_{i = 0}^{k - 1 - i} S^k B u_i
       where S is shift matrix
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def init_weights(self):
        super().init_weights()  # Initializes skip connection, B, C matrices
        # A column initialized in super().init_weights(), but now we zero-out
        a = torch.zeros(self.n_kernels, self.kernel_dim)
        self.register("a", a, trainable=False, lr=None, wd=None)
        
        # B Matrix - make it not learnable by default
        b = torch.zeros(self.n_kernels, self.kernel_dim)
        b[:, 0] = 1.
        self.register("b", b, trainable=False, lr=None, wd=None)
        
        # C matrix
        c = self.init_kernel_weights(self.kernel_init)
        self.register("c", c, trainable=True, lr=None, wd=None)
    
    def forward(self, u):
        return super().forward(u)
