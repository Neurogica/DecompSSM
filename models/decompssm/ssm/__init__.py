from .closed_loop.companion import ClosedLoopCompanionSSM
from .closed_loop.shift import ClosedLoopShiftSSM
from .companion import CompanionSSM
from .deep_decomposition import ClosedLoopDeepSSMDecomposition, DeepSSMDecomposition
from .dual import ClosedLoopDualSSM, DualSSM
from .freq_selective import ClosedLoopFrequencySelectiveSSM, FrequencySelectiveSSM
from .rotational_seasonal import ClosedLoopRotationalSeasonalSSM, RotationalSeasonalSSM
from .seasonal import ClosedLoopSeasonalSSM, SeasonalSSM
from .seasonal_diag import ClosedLoopDiagonalSeasonalSSM, DiagonalSeasonalSSM
from .selective_companion import ClosedLoopSelectiveCompanionSSM
from .shift import ShiftSSM


def init_ssm(config):
    supported_methods = [
        "companion",
        "closed_loop_companion",
        "shift",
        "closed_loop_shift",
        "seasonal_rotation",
        "closed_loop_seasonal_rotation",
        "seasonal_diagonal",
        "closed_loop_seasonal_diagonal",
        "rotational_seasonal",
        "closed_loop_rotational_seasonal",
        "closed_loop_selective_companion",
        "freq_selective_seasonal",
        "closed_loop_freq_selective_seasonal",
        "dual",
        "closed_loop_dual",
        "deep_decomposition",
        "closed_loop_deep_decomposition",
    ]
    if config["method"] == "companion":
        ssm = CompanionSSM
    elif config["method"] == "closed_loop_companion":
        ssm = ClosedLoopCompanionSSM
    elif config["method"] == "shift":
        ssm = ShiftSSM
    elif config["method"] == "closed_loop_shift":
        ssm = ClosedLoopShiftSSM
    elif config["method"] == "seasonal_rotation":
        ssm = SeasonalSSM
    elif config["method"] == "closed_loop_seasonal_rotation":
        ssm = ClosedLoopSeasonalSSM
    elif config["method"] == "seasonal_diagonal":
        ssm = DiagonalSeasonalSSM
    elif config["method"] == "closed_loop_seasonal_diagonal":
        ssm = ClosedLoopDiagonalSeasonalSSM
    elif config["method"] == "rotational_seasonal":
        ssm = RotationalSeasonalSSM
    elif config["method"] == "closed_loop_rotational_seasonal":
        ssm = ClosedLoopRotationalSeasonalSSM
    elif config["method"] == "closed_loop_selective_companion":
        ssm = ClosedLoopSelectiveCompanionSSM
    elif config["method"] == "freq_selective_seasonal":
        ssm = FrequencySelectiveSSM
    elif config["method"] == "closed_loop_freq_selective_seasonal":
        ssm = ClosedLoopFrequencySelectiveSSM
    elif config["method"] == "dual":
        ssm = DualSSM
    elif config["method"] == "closed_loop_dual":
        ssm = ClosedLoopDualSSM
    elif config["method"] == "deep_decomposition":
        ssm = DeepSSMDecomposition
    elif config["method"] == "closed_loop_deep_decomposition":
        ssm = ClosedLoopDeepSSMDecomposition
    else:
        raise NotImplementedError(f"SSM config method {config['method']} not implemented! Please choose from {supported_methods}")
    return ssm(**config["kwargs"])
