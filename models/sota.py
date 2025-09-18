import torch
import torch.nn.functional as F
from torch import nn

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer

"""
STFT which does mean

linear

poor

twoformer

"""


class GatedFusionModule(nn.Module):
    def __init__(self, channels):
        super(GatedFusionModule, self).__init__()
        self.channels = channels

        # 门控机制
        self.gate = nn.Sequential(nn.Linear(2 * channels, channels), nn.Sigmoid())

    def forward(self, dec_out, whole_out):
        # 将dec_out和whole_out沿通道维度连接
        concatenated = torch.cat((dec_out, whole_out), dim=-1)

        # 应用门控
        gate_scores = self.gate(concatenated)

        # 使用门控调整dec_out和whole_out
        gated_dec = gate_scores * dec_out
        gated_whole = (1 - gate_scores) * whole_out

        # 使用多头注意力进一步融合
        fusion_out = gated_dec + gated_whole

        return fusion_out


class GatedFusionModule(nn.Module):
    def __init__(self, channels, num_heads):
        super(GatedFusionModule, self).__init__()
        self.channels = channels
        self.num_heads = num_heads

        # 多头注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

        # 门控机制
        self.gate = nn.Sequential(nn.Linear(2 * channels, channels), nn.Sigmoid())

    def forward(self, dec_out, whole_out):
        # 将dec_out和whole_out沿通道维度连接
        concatenated = torch.cat((dec_out, whole_out), dim=-1)

        # 应用门控
        gate_scores = self.gate(concatenated)

        # 使用门控调整dec_out和whole_out
        gated_dec = gate_scores * dec_out
        gated_whole = (1 - gate_scores) * whole_out

        # 使用多头注意力进一步融合
        fusion_out, _ = self.attention(gated_dec, gated_whole, gated_whole)

        return fusion_out


class FusionLayer(nn.Module):
    def __init__(self, channel_size):
        super(FusionLayer, self).__init__()
        # 初始化权重生成网络，这里用一个简单的全连接层
        self.weights_net = nn.Sequential(nn.Linear(channel_size * 2, channel_size * 2), nn.ReLU(), nn.Linear(channel_size * 2, 2), nn.Softmax(dim=1))

    def forward(self, x1, x2):
        # 连接特征
        concatenated = torch.cat((x1, x2), dim=1)
        # 计算权重
        weights = self.weights_net(concatenated)
        # 应用权重
        weighted_sum = x1 * weights[:, :1, None] + x2 * weights[:, 1:, None]
        return weighted_sum


class ComplexModel(nn.Module):
    def __init__(self, channel_size=512):
        super(ComplexModel, self).__init__()
        self.encoder = nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, padding=1)
        self.decoder_head = nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, padding=1)
        self.whole_net = nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, padding=1)
        self.fusion_layer = FusionLayer(channel_size)

    def forward(self, x):
        # 假设x是输入特征图
        enc_out = self.encoder(x)
        dec_out = self.decoder_head(enc_out)
        whole_out = self.whole_net(enc_out)

        # 融合层
        final_out = self.fusion_layer(dec_out, whole_out)
        return final_out


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


def STFT_for_Period(x, k=2, n_fft=16, hop_length=8, win_length=16, window="hamming"):
    # Window setup based on user selection
    if window == "hann":
        window_tensor = torch.hann_window(win_length, periodic=True).to(x.device)
    elif window == "hamming":
        window_tensor = torch.hamming_window(win_length, periodic=True).to(x.device)
    else:
        raise ValueError(f"Unsupported window type: {window}")

    B, T, C = x.shape
    stft_results = []

    # Perform STFT for each channel separately
    for c in range(C):
        single_channel_data = x[:, :, c]
        stft_result = torch.stft(single_channel_data, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window_tensor, return_complex=True)
        stft_results.append(stft_result)

    stft_results = torch.stack(stft_results, dim=-1)
    xf_magnitude = torch.abs(stft_results).mean(dim=0)

    # Calculate frequency list
    frequency_list = xf_magnitude.mean(dim=-1).T
    frequency_list[:, 0] = 0  # Eliminate the DC component

    top_values = []
    for t in range(frequency_list.shape[0]):  # Iterate over time bins
        top_k_values, _ = torch.topk(frequency_list[t, :], 1)
        top_values.append(top_k_values)

    top_values = torch.tensor(top_values)
    k_amplitude, k_index = torch.topk(top_values, k)

    # Calculate period list
    period_list = T // k_amplitude

    # Calculate period weights
    period_weight = torch.abs(stft_results).mean(-1).permute(0, 2, 1)[:, k_index, :]
    period_weight = period_weight.mean(dim=-1)  # BKFmeanF
    # period_weight, _ = torch.max(period_weight, dim=2) # BKF max F
    return period_list, period_weight


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


##########full#############
def optimized_linear_interpolation_padding(x, seq_len, period, mode="linear"):
    B, C, T = x.shape
    # 计算需要的总长度，使其为周期的整数倍
    if seq_len % period != 0:
        length = ((seq_len // period) + 1) * period

    else:
        length = seq_len
        return x, length  # 如果已经是周期的整数倍，直接返回
        # 获取最后一个周期的数据
    last_period_start = seq_len - (seq_len % period)

    remaining_length = seq_len - last_period_start
    if remaining_length < 5:
        x = x[:, :, :last_period_start]

        length = length - 1
        return x, length
    last_period_data = x

    target = length
    # 为最后一个不完整的周期进行插值
    # 首先，我们需要调整last_period_data的形状以适应interpolate函数
    # interpolate函数期望的输入形状是[batch_size, channels, length]，其中channels在此处为C
    if mode == "linear":
        # 目标长度为一个完整的周期长度
        interpolated_data = F.interpolate(last_period_data, size=target, mode="linear", align_corners=False)
    elif mode == "nearest":
        interpolated_data = F.interpolate(last_period_data, size=target, mode="nearest")
    else:
        last_period_data_reshaped = last_period_data.unsqueeze(1)  # 添加一个维度作为channel维
        if mode == "bicubic":
            # 使用bicubic插值将图像的大小调整到48x48
            interpolated_data = F.interpolate(last_period_data_reshaped, size=(C, target), mode="bicubic", align_corners=True)
        elif mode == "bilinear":
            # 使用bicubic插值将图像的大小调整到48x48
            interpolated_data = F.interpolate(last_period_data_reshaped, size=(C, target), mode="bilinear", align_corners=True)

        interpolated_data = interpolated_data.squeeze(1)

    # 将插值后的数据重塑回原始维度 [batch_size, C, new_seq_len]

    return interpolated_data, length


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.k = configs.top_k
        self.patchH = configs.patchH
        self.patchW = configs.patchW
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # Encoder
        self.value_embedding = nn.Linear(self.patchH * self.patchW, configs.d_model)
        self.patch_num = ((4 - self.patchH) + 1) * 11
        self.encoder1 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.encoder2 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # Decoder
        if self.task_name == "long_term_forecast" or self.task_name == "short_term_forecast":
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "imputation":
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == "anomaly_detection":
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        self.head_nf = self.patch_num * 512
        if self.task_name == "long_term_forecast" or self.task_name == "short_term_forecast":
            self.head = FlattenHead(configs.enc_in, self.head_nf, 512, head_dropout=configs.dropout)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 24))
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        B, T, C = x_enc.shape
        # Embedding
        whole_out = self.enc_embedding(x_enc, x_mark_enc)
        whole_out, attns = self.encoder(whole_out, attn_mask=None)
        whole_out = whole_out[:, :C, :]

        period_list, period_weight = STFT_for_Period(x_enc, self.k)

        x = x_enc

        x = x.permute(0, 2, 1)

        res = []
        for i in range(self.k):
            period = period_list[i]
            period = int(period)
            x_padded, length = optimized_linear_interpolation_padding(x, self.seq_len, period)
            N_per = length // period
            out = x_padded.reshape(B, C, N_per, period).contiguous()
            out = self.adaptive_pool(out)
            res.append(out)
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, C, 4, 24, 1)
        res = torch.sum(res * period_weight, -1)

        patch_size = (self.patchH, self.patchW)

        stride = (1, 2)
        padding = (1, 1, 1, 1)  # (left, right, top, bottom)

        patches = res.unfold(2, patch_size[0], stride[0]).unfold(3, patch_size[1], stride[1])

        x = patches.contiguous().view(B, C, -1, patch_size[0] * patch_size[1])

        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # reshape back
        enc_out = self.value_embedding(x)

        enc_out, attns = self.encoder1(enc_out, attn_mask=None)
        enc_out = torch.reshape(enc_out, (-1, C, enc_out.shape[-2], enc_out.shape[-1]))

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]

        final_out = dec_out + whole_out

        dec_out = self.projection(final).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "long_term_forecast" or self.task_name == "short_term_forecast":
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
        # res = []
        # for i in range(self.k):
        #     period = period_list[i]
        #     # padding
        #     if (self.seq_len + self.pred_len) % period != 0:
        #         length = (
        #                          ((self.seq_len + self.pred_len) // period) + 1) * period
        #         padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
        #         out = torch.cat([x, padding], dim=1)
        #     else:
        #         length = (self.seq_len + self.pred_len)
        #         out = x
        #     # reshape
        #     N_per = length // period
        #     out = out.reshape(B, N_per, period,
        #                       N).permute(0, 3, 1, 2).contiguous()
        #     # 2D conv: from 1d Variation to 2d Variation
        #     out = self.conv(out)
        #     # reshape back
        #     out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
        #     res.append(out[:, :(self.seq_len + self.pred_len), :])
        # res = torch.stack(res, dim=-1)
        # # adaptive aggregation
        # period_weight = F.softmax(period_weight, dim=1)
        # period_weight = period_weight.unsqueeze(
        #     1).unsqueeze(1).repeat(1, T, N, 1)
        # res = torch.sum(res * period_weight, -1)
        # # residual connection
        # res = res + x
