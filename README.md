# DecompSSM: A Decomposition-Based State Space Model for Multivariate Time-Series Forecasting

## 💡 About
This repository provides the implementation of **DecompSSM**, as presented in our paper:
*"A DECOMPOSITION-BASED STATE SPACE MODEL FOR MULTIVARIATE TIME-SERIES FORECASTING"*.

DecompSSM is an end-to-end decomposition framework using three parallel deep state space model branches to capture trend, seasonal, and residual components. The model features adaptive temporal scales via an input-dependent predictor, a refinement module for shared cross-variable context, and an auxiliary loss that enforces reconstruction and orthogonality.

## 🚀 Key Features
- **Component-wise Decomposition**: Parallel branches for trend, seasonal, and residual components
- **Adaptive Temporal Scales**: Input-dependent timescale adaptation via Adaptive Step Predictor (ASP)
- **Global Context Refinement**: Cross-variable context sharing and synchronization
- **Auxiliary Decomposition Loss**: Enforces reconstruction and orthogonality constraints

## 🛠️ Getting Started

### Installation
Clone the repository and set up the environment:
```bash
git clone https://github.com/Neurogica/DecompSSM.git
cd DecompSSM
uv venv
source .venv/bin/activate
uv sync
```

### Data Preparation
Download the datasets and place them in the `./dataset/` directory:
- **ECL**: Electricity Consuming Load dataset
- **Weather**: Weather monitoring dataset  
- **ETTm2**: Electricity Transforming Temperature dataset
- **PEMS04**: Traffic flow dataset

The directory structure should look like:
```bash
.
└── dataset
    ├── ECL/
    ├── weather/
    ├── ETT-small/
    └── PEMS/
```

## 🎯 Experiments

### Basic Usage
Run DecompSSM on different datasets using the following commands:

```bash
# ECL dataset
python run.py --task_name long_term_forecast --is_training 1 --model_id DecompSSM --model DecompSSM --data ECL --features M --seq_len 96 --pred_len 96

# Weather dataset  
python run.py --task_name long_term_forecast --is_training 1 --model_id DecompSSM --model DecompSSM --data weather --features M --seq_len 96 --pred_len 96

# ETTm2 dataset
python run.py --task_name long_term_forecast --is_training 1 --model_id DecompSSM --model DecompSSM --data ETTm2 --features M --seq_len 96 --pred_len 96

# PEMS04 dataset
python run.py --task_name long_term_forecast --is_training 1 --model_id DecompSSM --model DecompSSM --data PEMS04 --features M --seq_len 96 --pred_len 96
```


## 📊 Results
DecompSSM outperformed strong baselines across standard benchmarks:
- **ECL**: 28/32 best results across horizons and metrics
- **Weather**: Consistent improvements in MSE and MAE
- **ETTm2**: Strong performance on long-term forecasting
- **PEMS04**: Effective traffic flow prediction

## 📝 License
This work is licensed under the BSD-3-Clause-Clear License. See [LICENSE](LICENSE) for details.
