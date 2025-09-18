import argparse
import random

import numpy as np
import torch

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_imputation import Exp_Imputation
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from utils.print_args import print_args

if __name__ == "__main__":
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="TimesNet")

    # basic config
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        default="long_term_forecast",
        help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
    )
    parser.add_argument("--is_training", type=int, required=True, default=1, help="status")
    parser.add_argument("--model_id", type=str, required=True, default="test", help="model id")
    parser.add_argument("--model", type=str, required=True, default="Autoformer", help="model name, options: [Autoformer, Transformer, TimesNet]")

    # data loader
    parser.add_argument("--data", type=str, required=True, default="ETTm1", help="dataset type")
    parser.add_argument("--root_path", type=str, default="./data/ETT/", help="root path of the data file")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument("--target", type=str, default="OT", help="target feature in S or MS task")
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="location of model checkpoints")

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction sequence length")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly", help="subset for M4")
    parser.add_argument("--inverse", action="store_true", help="inverse output data", default=False)

    # inputation task
    parser.add_argument("--mask_rate", type=float, default=0.25, help="mask ratio")

    # anomaly detection task
    parser.add_argument("--anomaly_ratio", type=float, default=0.25, help="prior anomaly ratio (%)")

    # model define
    parser.add_argument("--expand", type=int, default=2, help="expansion factor for Mamba")
    parser.add_argument("--d_conv", type=int, default=4, help="conv kernel size for Mamba")
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument("--moving_avg", type=int, default=25, help="window size of moving average")
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil", action="store_false", help="whether to use distilling in encoder, using this argument means not using distilling", default=True
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--embed", type=str, default="timeF", help="time features encoding, options:[timeF, fixed, learned]")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument("--output_attention", action="store_true", help="whether to output attention in ecoder")
    parser.add_argument("--channel_independence", type=int, default=1, help="0: channel dependence 1: channel independence for FreTS model")
    parser.add_argument("--decomp_method", type=str, default="moving_avg", help="method of series decompsition, only support moving_avg or dft_decomp")
    parser.add_argument("--use_norm", type=int, default=1, help="whether to use normalize; True 1 False 0")
    parser.add_argument("--down_sampling_layers", type=int, default=3, help="num of down sampling layers")
    parser.add_argument("--down_sampling_window", type=int, default=2, help="down sampling window size")
    parser.add_argument("--down_sampling_method", type=str, default="conv", help="down sampling method, only support avg, max, conv")
    parser.add_argument("--seg_len", type=int, default=48, help="the length of segmen-wise iteration of SegRNN")

    # optimization
    parser.add_argument("--num_workers", type=int, default=10, help="data loader num workers")
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
    parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision training", default=False)

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multile gpus")

    # de-stationary projector params
    parser.add_argument("--p_hidden_dims", type=int, nargs="+", default=[128, 128], help="hidden layer dimensions of projector (List)")
    parser.add_argument("--p_hidden_layers", type=int, default=2, help="number of hidden layers in projector")

    # metrics (dtw)
    parser.add_argument(
        "--use_dtw", type=bool, default=False, help="the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)"
    )

    # Augmentation
    parser.add_argument("--augmentation_ratio", type=int, default=0, help="How many times to augment")
    parser.add_argument("--seed", type=int, default=2, help="Randomization seed")
    parser.add_argument("--jitter", default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument("--scaling", default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument("--permutation", default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument("--randompermutation", default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument("--magwarp", default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument("--timewarp", default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument("--windowslice", default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument("--windowwarp", default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument("--rotation", default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument("--spawner", default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument("--dtwwarp", default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument("--shapedtwwarp", default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument("--wdba", default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument("--discdtw", default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument("--discsdtw", default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument("--extra_tag", type=str, default="", help="Anything extra")

    # PPDformer specific arguments
    parser.add_argument("--patchH", type=int, default=2, help="patch height for PPDformer")
    parser.add_argument("--patchW", type=int, default=2, help="patch width for PPDformer")
    parser.add_argument("--strideH", type=int, default=1, help="stride height for PPDformer")
    parser.add_argument("--strideW", type=int, default=2, help="stride width for PPDformer")
    parser.add_argument("--normal", type=int, default=0, help="normalization for PPDformer (0 or 1)")

    # DecompSSM specific arguments
    parser.add_argument("--use_dynamic_merge_gate", action="store_true", help="use dynamic merge gate in dual SSM", default=False)
    parser.add_argument("--use_freq_selective_seasonal", action="store_true", help="use frequency-selective seasonal SSM", default=False)
    parser.add_argument("--merge_hidden_dim", type=int, default=None, help="hidden dimension for merge gate in dual SSM")
    parser.add_argument("--component_predictor", type=str, default="mlp", help="component predictor head: mlp, linear, or ssm")
    parser.add_argument("--use_residual_in_forecast", action="store_true", help="include residual component in forecast sum", default=False)
    parser.add_argument("--use_concat_prediction", action="store_true", help="concatenate components and use unified prediction head", default=False)

    # Normalization arguments
    parser.add_argument("--normalization", type=str, default="standard", help="normalization type: revin, standard, fan, san")
    parser.add_argument("--freq_topk", type=int, default=20, help="number of top frequency components for FAN normalization")
    parser.add_argument("--period_len", type=int, default=24, help="seasonal period length for SAN normalization")
    parser.add_argument("--station_type", type=str, default="adaptive", help="station type for SAN normalization")
    parser.add_argument("--norm_eps", type=float, default=1e-8, help="epsilon for normalization stability")
    parser.add_argument("--norm_affine", action="store_true", help="use affine transformation in normalization", default=True)
    parser.add_argument("--rfft", action="store_true", help="use real FFT for FAN normalization", default=True)

    # DeepSSMDecomposition specific arguments
    parser.add_argument("--d_state", type=int, default=16, help="state dimension for DeepSSMDecomposition")
    parser.add_argument("--n_decomp_layers", type=int, default=2, help="number of decomposition layers for DeepSSMDecomposition")
    parser.add_argument("--top_k_freqs", type=int, default=3, help="number of top frequencies to track for DeepSSMDecomposition")
    parser.add_argument("--freq_momentum", type=float, default=0.2, help="frequency EMA momentum for DeepSSMDecomposition")
    parser.add_argument("--fusion_method", type=str, default="learned", help="component fusion method for DeepSSMDecomposition (learned, attention, gate)")

    # DeepS5Decomposition specific arguments
    parser.add_argument("--s5_blocks", type=int, default=1, help="number of S5 blocks for DeepS5Decomposition")
    parser.add_argument("--state_size", type=int, default=64, help="S5 state dimension for DeepS5DecompositionV2")
    parser.add_argument("--s5_factor_rank", type=int, default=0, help="low-rank factorization rank (0 disables)")
    parser.add_argument("--s5_liquid", action="store_true", default=False, help="use liquid time-constant SSM")
    parser.add_argument("--s5_degree", type=int, default=1, help="input operator degree for S5")
    parser.add_argument("--model_type", type=str, default="s5", help="model type for DecompSSM")

    # S5 temporal dynamics parameters
    parser.add_argument("--trend_dt_min", type=float, default=0.01, help="trend component dt_min (slow dynamics)")
    parser.add_argument("--trend_dt_max", type=float, default=0.1, help="trend component dt_max (slow dynamics)")
    parser.add_argument("--seasonal_dt_min", type=float, default=0.001, help="seasonal component dt_min (balanced dynamics)")
    parser.add_argument("--seasonal_dt_max", type=float, default=0.01, help="seasonal component dt_max (balanced dynamics)")
    parser.add_argument("--residual_dt_min", type=float, default=0.0001, help="residual component dt_min (fast dynamics)")
    parser.add_argument("--residual_dt_max", type=float, default=0.001, help="residual component dt_max (fast dynamics)")

    # S5 bidirectional settings
    parser.add_argument("--trend_bidir", action="store_true", help="use bidirectional S5 for trend", default=False)
    parser.add_argument("--seasonal_bidir", action="store_true", help="use bidirectional S5 for seasonal", default=True)
    parser.add_argument("--residual_bidir", action="store_true", help="use bidirectional S5 for residual", default=False)

    # Predictor SSM (if --component_predictor ssm)
    parser.add_argument("--pred_s5_blocks", type=int, default=1, help="S5 blocks in predictor head")
    parser.add_argument("--pred_s5_factor_rank", type=int, default=0, help="S5 low-rank rank in predictor head (0 disables)")
    parser.add_argument("--pred_s5_liquid", action="store_true", default=False, help="use liquid SSM in predictor head")
    parser.add_argument("--pred_s5_degree", type=int, default=1, help="predictor S5 input operator degree")

    # Channel interaction parameters
    parser.add_argument("--channel_interaction_strength", type=float, default=0.1, help="learnable channel interaction strength")
    parser.add_argument("--use_channel_interaction", action="store_true", help="enable channel interaction", default=True)

    # TimeMixerPP specific arguments
    parser.add_argument("--channel_mixing", type=int, default=0, help="channel mixing for TimeMixerPP (0: False, 1: True)")

    # Loss function weights and parameters
    parser.add_argument("--aux_loss_weight", type=float, default=0.1, help="auxiliary loss weight")
    parser.add_argument("--variance_reg_weight", type=float, default=0.1, help="variance regularization weight")
    parser.add_argument("--temporal_consistency_weight", type=float, default=0.05, help="temporal consistency weight")
    parser.add_argument("--orthogonality_weight", type=float, default=0.1, help="component orthogonality weight")
    parser.add_argument("--spatial_smoothness_weight", type=float, default=0.03, help="spatial smoothness weight")
    parser.add_argument("--diversity_reg_weight", type=float, default=0.02, help="channel diversity regularization weight")

    # Additional auxiliary loss arguments for DeepSSMDecomposition (legacy)
    parser.add_argument("--lambda_recon", type=float, default=1.0, help="reconstruction loss weight for DeepSSMDecomposition (legacy)")
    parser.add_argument("--lambda_frequency", type=float, default=0.2, help="frequency consistency loss weight for DeepS5DecompositionV2 (legacy)")
    parser.add_argument("--lambda_freq", type=float, default=2.0, help="frequency constraint weight for DeepSSMDecomposition")
    parser.add_argument("--lambda_sparsity", type=float, default=0.1, help="sparsity penalty weight for DeepSSMDecomposition")

    # Theoretically grounded decomposition loss arguments for DeepS5DecompositionV2
    parser.add_argument("--lambda_reconstruction", type=float, default=1.0, help="perfect reconstruction loss weight (fundamental decomposition constraint)")
    parser.add_argument("--lambda_orthogonality", type=float, default=0.1, help="component orthogonality loss weight (statistical independence principle)")
    parser.add_argument("--lambda_sparsity_prior", type=float, default=0.05, help="sparsity prior loss weight (information-theoretic component separation)")

    # Trial name for custom experiment naming and logging
    parser.add_argument("--trial_name", type=str, default="", help="custom trial name for experiment identification and logging")

    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print_args(args)

    Exp: type[Exp_Long_Term_Forecast | Exp_Short_Term_Forecast | Exp_Imputation | Exp_Anomaly_Detection | Exp_Classification]
    if args.task_name == "long_term_forecast":
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == "short_term_forecast":
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == "imputation":
        Exp = Exp_Imputation
    elif args.task_name == "anomaly_detection":
        Exp = Exp_Anomaly_Detection
    elif args.task_name == "classification":
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            trial_suffix = f"_{args.trial_name}" if args.trial_name else ""
            setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_expand{args.expand}_dc{args.d_conv}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}{trial_suffix}_{ii}"

            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)

            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        trial_suffix = f"_{args.trial_name}" if args.trial_name else ""
        setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_expand{args.expand}_dc{args.d_conv}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}{trial_suffix}_{ii}"

        exp = Exp(args)  # set experiments
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
