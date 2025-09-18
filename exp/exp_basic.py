import os
from typing import Any

import torch

# Import all models from models package (conditional imports handled in __init__.py)
from models import (
    MICN,
    Autoformer,
    Crossformer,
    DecompSSM,
    DecompSSMV2,
    DeepMLPDecomposition,
    DeepS5Decomposition,
    DeepS5DecompositionV2,
    DeepSSMDecomposition,
    DLinear,
    ETSformer,
    FEDformer,
    FiLM,
    FreTS,
    Informer,
    Koopa,
    LightTS,
    Mamba,
    MambaSimple,
    Nonstationary_Transformer,
    PatchTST,
    PPDformer,
    Pyraformer,
    Reformer,
    SegRNN,
    Spacetime,
    TemporalFusionTransformer,
    TiDE,
    TimeMixer,
    TimeMixerPP,
    TimesNet,
    Transformer,
    TSMixer,
    iTransformer,
)


class Exp_Basic:
    def __init__(self, args: Any) -> None:
        self.args = args
        self.model_dict = {
            "TimesNet": TimesNet,
            "Autoformer": Autoformer,
            "Transformer": Transformer,
            "Nonstationary_Transformer": Nonstationary_Transformer,
            "DLinear": DLinear,
            "FEDformer": FEDformer,
            "Informer": Informer,
            "LightTS": LightTS,
            "Reformer": Reformer,
            "ETSformer": ETSformer,
            "PatchTST": PatchTST,
            "PPDformer": PPDformer,
            "Pyraformer": Pyraformer,
            "MICN": MICN,
            "Crossformer": Crossformer,
            "FiLM": FiLM,
            "iTransformer": iTransformer,
            "Koopa": Koopa,
            "TiDE": TiDE,
            "FreTS": FreTS,
            "MambaSimple": MambaSimple,
            "Mamba": Mamba,
            "TimeMixer": TimeMixer,
            "TimeMixerPP": TimeMixerPP,
            "TSMixer": TSMixer,
            "SegRNN": SegRNN,
            "TemporalFusionTransformer": TemporalFusionTransformer,
            "Spacetime": Spacetime,
            "DecompSSM": DecompSSM,
            "DecompSSMV2": DecompSSMV2,
            "DeepSSMDecomposition": DeepSSMDecomposition,
            "DeepMLPDecomposition": DeepMLPDecomposition,
            "DeepS5Decomposition": DeepS5Decomposition,
            "DeepS5DecompositionV2": DeepS5DecompositionV2,
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self) -> Any:
        raise NotImplementedError

    def _acquire_device(self) -> torch.device:
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self) -> Any:
        pass

    def vali(self) -> Any:
        pass

    def train(self) -> Any:
        pass

    def test(self) -> Any:
        pass
