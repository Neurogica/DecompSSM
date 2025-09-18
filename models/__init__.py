# Import all models to make them available
from .Autoformer import Model as Autoformer
from .Crossformer import Model as Crossformer
from .DecompSSM import Model as DecompSSM
from .DecompSSM import Model as DecompSSMV2
from .DeepMLPDecomposition import Model as DeepMLPDecomposition
from .DeepS5Decomposition import Model as DeepS5Decomposition
from .DeepS5DecompositionV2 import Model as DeepS5DecompositionV2
from .DeepSSMDecomposition import Model as DeepSSMDecomposition
from .DLinear import Model as DLinear
from .ETSformer import Model as ETSformer
from .FEDformer import Model as FEDformer
from .FiLM import Model as FiLM
from .FreTS import Model as FreTS
from .Informer import Model as Informer
from .iTransformer import Model as iTransformer
from .Koopa import Model as Koopa
from .LightTS import Model as LightTS
from .Mamba import Model as Mamba
from .MambaSimple import Model as MambaSimple
from .MICN import Model as MICN
from .Nonstationary_Transformer import Model as Nonstationary_Transformer
from .PatchTST import Model as PatchTST
from .PPDformer import Model as PPDformer
from .Pyraformer import Model as Pyraformer
from .Reformer import Model as Reformer
from .SegRNN import Model as SegRNN

# Make Spacetime import optional to avoid blocking other models
try:
    from .SpacetimeWrapper import Model as Spacetime
except ImportError:
    Spacetime = None

from .TemporalFusionTransformer import Model as TemporalFusionTransformer
from .TiDE import Model as TiDE
from .TimeMixer import Model as TimeMixer
from .TimeMixerPP import Model as TimeMixerPP
from .TimesNet import Model as TimesNet
from .Transformer import Model as Transformer
from .TSMixer import Model as TSMixer

__all__ = [
    "Autoformer",
    "Crossformer",
    "DecompSSM",
    "DecompSSMV2",
    "DeepSSMDecomposition",
    "DeepMLPDecomposition",
    "DeepS5Decomposition",
    "DeepS5DecompositionV2",
    "DLinear",
    "ETSformer",
    "FEDformer",
    "FiLM",
    "FreTS",
    "Informer",
    "Koopa",
    "LightTS",
    "Mamba",
    "MambaSimple",
    "MICN",
    "Nonstationary_Transformer",
    "PatchTST",
    "PPDformer",
    "Pyraformer",
    "Reformer",
    "SegRNN",
    "TemporalFusionTransformer",
    "TiDE",
    "TimeMixer",
    "TimeMixerPP",
    "TimesNet",
    "Transformer",
    "TSMixer",
    "iTransformer",
    "Spacetime",
]
