# =========================================================
# Paimon: Filtering Patch-Induced Redundancy
# Attention-Replacement Patch Filtering Version
#
# Model name: Paimon
#
# Main structure:
#   Input [B, L, C]
#     -> RevIN
#     -> Channel-independent reshape [B*C, L]
#     -> Patch-Aware Spectral Embedding
#     -> Paimon Encoder
#          Period-aware Patch Filter
#          Filter1D / Attention
#          FFN
#     -> Forecast Head
#     -> RevIN denorm
#
# Main update:
#   Keep the original period-aware patch filtering structure.
#   Only infer filter_kernels from period in Paimon.__init__.
#
# Effective raw receptive field:
#   R(k) ~= patch_len + (k - 1) * stride
#
# Period-aware patch filter kernels:
#   short: k = 3
#   mid:   approximately half-period coverage
#   long:  approximately full-period coverage
#
# If filter_kernels is manually specified, use it directly.
# If filter_kernels is None / "auto", infer from period.
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Utility
# =========================================================
def _to_bool(v):
    if isinstance(v, bool):
        return v

    if isinstance(v, (int, float)):
        return v != 0

    if isinstance(v, str):
        return v.strip().lower() in ["1", "true", "yes", "y", "on"]

    return bool(v)


def _parse_optional_float(v):
    if v is None:
        return None

    if isinstance(v, str):
        if v.strip().lower() in ["none", "null", "", "auto"]:
            return None
        return float(v)

    return float(v)


def _parse_kernels(v):
    """
    Parse filter_kernels.

    Supported:
        None        -> None
        "auto"      -> None
        "3,5,7"     -> (3, 5, 7)
        (3, 5, 7)   -> (3, 5, 7)
        [3, 5, 7]   -> (3, 5, 7)
    """

    if v is None:
        return None

    if isinstance(v, str):
        v_clean = v.strip().lower()

        if v_clean in ["none", "null", "", "auto"]:
            return None

        kernels = tuple(
            int(k.strip()) for k in v.split(",") if k.strip() != ""
        )
    else:
        kernels = tuple(int(k) for k in v)

    if len(kernels) == 0:
        return None

    for k in kernels:
        if k <= 0:
            raise ValueError(f"Kernel size must be positive, got {k}.")
        if k % 2 == 0:
            raise ValueError(f"Kernel size should be odd, got {k}.")

    return kernels


def infer_patch_filter_kernels(
    patch_num,
    patch_len,
    stride,
    period=24,
    period_rf_scale=1.0,
    target_num=3
):
    """
    Infer patch filter kernels from period.

    Effective raw receptive field:
        R(k) ~= patch_len + (k - 1) * stride

    Try to return three kernels:
        1. short filter kernel: 3
        2. mid filter kernel:   half-period scale
        3. long filter kernel:  full-period scale

    If period-derived kernels duplicate each other, fill with 5, 7, 9 ...
    while respecting patch_num.

    Args:
        patch_num:
            Number of patch tokens.

        patch_len:
            Patch length in raw time steps.

        stride:
            Patch stride in raw time steps.

        period:
            Period in raw time steps.

        period_rf_scale:
            Receptive-field expansion factor applied to the
            token-level receptive-field offset after raw-period
            alignment. It does not change the raw period itself.
            period_rf_scale > 1 enlarges the inferred token
            expansion, while period_rf_scale < 1 makes it more
            conservative.

        target_num:
            Target number of patch filter kernels.

    Return:
        tuple[int]
    """

    patch_num = int(patch_num)
    patch_len = int(patch_len)
    stride = int(stride)

    if period is None:
        period = 24

    period = float(period)
    period_rf_scale = float(period_rf_scale)

    if period_rf_scale <= 0:
        raise ValueError(
            f"period_rf_scale must be positive, got {period_rf_scale}."
        )

    if patch_num <= 0:
        raise ValueError(f"Invalid patch_num={patch_num}.")

    if patch_len <= 0:
        raise ValueError(f"Invalid patch_len={patch_len}.")

    if stride <= 0:
        raise ValueError(f"Invalid stride={stride}.")

    if target_num <= 0:
        raise ValueError(f"Invalid target_num={target_num}.")

    # Maximum useful odd kernel.
    # Avoid kernels much larger than token length.
    max_k = patch_num if patch_num % 2 == 1 else patch_num - 1
    max_k = max(3, max_k)

    def to_odd(k):
        k = int(round(float(k)))

        if k < 3:
            k = 3

        if k % 2 == 0:
            k += 1

        if k > max_k:
            k = max_k

        if k % 2 == 0:
            k -= 1

        return max(3, k)

    kernels = []

    def add_kernel(k):
        k = to_odd(k)
        if k not in kernels:
            kernels.append(k)

    # 1. Short patch-filter branch.
    add_kernel(3)

    # 2. Half-period scale.
    #    The scaling factor controls the token-level expansion
    #    after raw-time period alignment, rather than modifying
    #    the raw period itself.
    k_half = period_rf_scale * ((0.5 * period - patch_len) / stride) + 1
    add_kernel(k_half)

    # 3. Full-period scale.
    k_full = period_rf_scale * ((1.0 * period - patch_len) / stride) + 1
    add_kernel(k_full)

    # 4. Fill missing kernels to keep multi-scale capacity.
    candidate = 5
    while len(kernels) < target_num and candidate <= max_k:
        add_kernel(candidate)
        candidate += 2

    return tuple(sorted(kernels[:target_num]))


# =========================================================
# RevIN
# =========================================================
class RevIN(nn.Module):
    """
    Reversible Instance Normalization.

    Input:
        x: [B, L, C]
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        affine=True,
        subtract_last=False
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.gamma = None
            self.beta = None

        self.mean = None
        self.std = None
        self.last = None

    def forward(self, x, mode: str):
        if mode == "norm":
            if self.subtract_last:
                self.last = x[:, -1:, :].detach()
                x_centered = x - self.last
                self.mean = None
            else:
                self.mean = x.mean(dim=1, keepdim=True).detach()
                x_centered = x - self.mean
                self.last = None

            self.std = (
                x_centered.std(dim=1, keepdim=True, unbiased=False).detach()
                + self.eps
            )

            x = x_centered / self.std

            if self.affine:
                x = x * self.gamma + self.beta

            return x

        elif mode == "denorm":
            if self.std is None:
                raise RuntimeError("RevIN denorm called before norm.")

            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)

            x = x * self.std

            if self.subtract_last:
                if self.last is None:
                    raise RuntimeError("RevIN last value is missing.")
                x = x + self.last
            else:
                if self.mean is None:
                    raise RuntimeError("RevIN mean is missing.")
                x = x + self.mean

            return x

        else:
            raise ValueError("mode must be 'norm' or 'denorm'.")


# =========================================================
# Standard Patch Embedding
# Used for ablation: w/o Patch-Aware Spectral Embedding
# =========================================================
class StandardPatchEmbedding(nn.Module):
    """
    Standard time-domain patch embedding.

    Input:
        x: [B, L]

    Output:
        tokens: [B, N, D]
    """

    def __init__(self, patch_len, stride, d_model, dropout=0.0):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        self.patch_proj = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride,
            padding=0,
            bias=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, L]
        """

        tokens = self.patch_proj(x.unsqueeze(1))
        # [B, D, N]

        tokens = tokens.transpose(1, 2).contiguous()
        # [B, N, D]

        tokens = self.norm(tokens)
        tokens = self.drop(tokens)

        return tokens


# =========================================================
# Patch-Aware Spectral Embedding
# =========================================================
class PatchAwareSpectralEmbedding(nn.Module):
    """
    Patch-Aware Spectral Embedding.

    Core idea:
        Do not split the input into low/mid/high spectral branches.
        Do not perform patch-level spectral decomposition.

        Instead:
            raw sequence
                -> rFFT over full input length
                -> learnable frequency bias + adaptive spectral gate
                -> irFFT
                -> residual modulation
                -> optional patch norm
                -> Conv1D patch embedding

    Input:
        x: [B, L]

    Output:
        tokens: [B, N, D]
    """

    def __init__(
        self,
        seq_len,
        patch_len,
        stride,
        d_model,
        dropout=0.0,
        low_ratio=0.35,
        mid_ratio=0.70,
        gate_hidden_dim=None,
        use_adaptive_gate=True,
        use_learnable_freq_bias=True,
        residual_scale_init=-1.0,
        gate_temperature=1.0,
        use_patch_norm=True,
        eps=1e-6
    ):
        super().__init__()

        if not (0.0 < low_ratio < mid_ratio < 1.0):
            raise ValueError(
                "Require 0 < low_ratio < mid_ratio < 1. "
                f"Got low_ratio={low_ratio}, mid_ratio={mid_ratio}."
            )

        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive.")

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        self.low_ratio = low_ratio
        self.mid_ratio = mid_ratio

        self.use_adaptive_gate = use_adaptive_gate
        self.use_learnable_freq_bias = use_learnable_freq_bias
        self.gate_temperature = gate_temperature
        self.use_patch_norm = use_patch_norm
        self.eps = eps

        self.freq_len = seq_len // 2 + 1
        self.patch_num = (seq_len - patch_len) // stride + 1

        if self.patch_num <= 0:
            raise ValueError(
                f"Invalid patch setting: seq_len={seq_len}, "
                f"patch_len={patch_len}, stride={stride}."
            )

        self.patch_proj = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride,
            padding=0,
            bias=True
        )

        if use_learnable_freq_bias:
            self.freq_bias = nn.Parameter(torch.zeros(self.freq_len))
        else:
            self.freq_bias = None

        self.stat_dim = 7

        if gate_hidden_dim is None:
            gate_hidden_dim = max(16, d_model // 4)

        if use_adaptive_gate:
            self.stat_norm = nn.LayerNorm(self.stat_dim)
            self.adaptive_gate = nn.Sequential(
                nn.Linear(self.stat_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden_dim, self.freq_len)
            )
        else:
            self.stat_norm = None
            self.adaptive_gate = None

        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _spectral_entropy(self, power):
        """
        power: [B, F]

        return:
            entropy: [B, 1]
        """

        prob = power / power.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        entropy = -(prob * torch.log(prob + self.eps)).sum(
            dim=-1,
            keepdim=True
        )

        norm = torch.log(
            torch.tensor(
                float(power.shape[-1]),
                device=power.device,
                dtype=power.dtype
            ) + self.eps
        )

        entropy = entropy / norm.clamp_min(self.eps)

        return entropy

    def _build_stats(self, power):
        """
        power: [B, F]

        return:
            stats: [B, 7]
        """

        B, F_len = power.shape

        low_end = max(1, int(F_len * self.low_ratio))
        mid_end = max(low_end + 1, int(F_len * self.mid_ratio))
        mid_end = min(mid_end, F_len)

        low_energy = power[:, :low_end].mean(dim=-1, keepdim=True)

        if mid_end > low_end:
            mid_energy = power[:, low_end:mid_end].mean(
                dim=-1,
                keepdim=True
            )
        else:
            mid_energy = torch.zeros_like(low_energy)

        if mid_end < F_len:
            high_energy = power[:, mid_end:].mean(dim=-1, keepdim=True)
        else:
            high_energy = torch.zeros_like(low_energy)

        band_energy = torch.cat(
            [low_energy, mid_energy, high_energy],
            dim=-1
        )
        # [B, 3]

        total_band = band_energy.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        band_ratio = band_energy / total_band
        # [B, 3]

        total_energy = torch.log1p(power.mean(dim=-1, keepdim=True))
        entropy = self._spectral_entropy(power)

        high_to_low = high_energy / low_energy.clamp_min(self.eps)
        mid_to_low = mid_energy / low_energy.clamp_min(self.eps)

        high_to_low = torch.log1p(high_to_low)
        mid_to_low = torch.log1p(mid_to_low)

        stats = torch.cat(
            [
                total_energy,
                band_ratio,
                entropy,
                high_to_low,
                mid_to_low
            ],
            dim=-1
        )
        # [B, 7]

        return stats

    def _make_spectral_gate(self, x_freq):
        """
        x_freq: [B, F], complex

        return:
            gate: [B, F]
        """

        power = x_freq.abs().pow(2).clamp_min(self.eps)
        # [B, F]

        logits = torch.zeros_like(power)

        if self.use_learnable_freq_bias:
            logits = logits + self.freq_bias.view(1, -1).to(
                device=power.device,
                dtype=power.dtype
            )

        if self.use_adaptive_gate:
            stats = self._build_stats(power)

            stats = stats.to(dtype=self.stat_norm.weight.dtype)
            stats = self.stat_norm(stats)

            adaptive_logits = self.adaptive_gate(stats)
            adaptive_logits = adaptive_logits.to(
                device=power.device,
                dtype=power.dtype
            )

            logits = logits + adaptive_logits

        gate = torch.sigmoid(logits / self.gate_temperature)
        # [B, F]

        return gate

    def forward(self, x):
        """
        x: [B, L]
        """

        B, L = x.shape

        if L != self.seq_len:
            raise ValueError(
                f"Input length {L} does not match seq_len={self.seq_len}."
            )

        orig_dtype = x.dtype

        fft_input = (
            x.float()
            if x.dtype in [torch.float16, torch.bfloat16]
            else x
        )

        x_freq = torch.fft.rfft(fft_input, dim=-1, norm="ortho")
        # [B, F]

        gate = self._make_spectral_gate(x_freq)
        # [B, F]

        gated_freq = x_freq * gate.to(dtype=x_freq.real.dtype)
        # [B, F]

        x_filtered = torch.fft.irfft(
            gated_freq,
            n=self.seq_len,
            dim=-1,
            norm="ortho"
        )
        # [B, L]

        x_filtered = x_filtered.to(dtype=orig_dtype)

        scale = torch.sigmoid(self.residual_scale)

        x_mod = x + scale * (x_filtered - x)
        # [B, L]

        if self.use_patch_norm:
            x_mod = x_mod - x_mod.mean(dim=1, keepdim=True)
            x_mod = x_mod / (
                x_mod.std(dim=1, keepdim=True, unbiased=False) + self.eps
            )

        tokens = self.patch_proj(x_mod.unsqueeze(1))
        # [B, D, N]

        tokens = tokens.transpose(1, 2).contiguous()
        # [B, N, D]

        tokens = self.norm(tokens)
        tokens = self.drop(tokens)

        return tokens


# =========================================================
# Filter1D for Patch Tokens
# =========================================================
class Filter1D(nn.Module):
    """
    Fourier-domain token filtering with learnable complex filter and frequency gate.

    Input:
        x: [B, N, D]

    Output:
        out: [B, N, D]
    """

    def __init__(
        self,
        d_model,
        seq_len,
        filter_init_scale=0.02,
        dropout=0.1,
        use_filter_gate=True
    ):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.freq_len = seq_len // 2 + 1
        self.use_filter_gate = use_filter_gate

        self.weight_real = nn.Parameter(
            filter_init_scale * torch.randn(self.freq_len, d_model)
        )
        self.weight_imag = nn.Parameter(
            filter_init_scale * torch.randn(self.freq_len, d_model)
        )

        if use_filter_gate:
            self.freq_gate = nn.Parameter(torch.zeros(self.freq_len, d_model))
        else:
            self.freq_gate = None

        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _resize_weight(self, weight, target_freq_len):
        """
        weight: [F, D]
        """

        if target_freq_len == weight.shape[0]:
            return weight

        weight = weight.transpose(0, 1).unsqueeze(0)
        # [1, D, F]

        weight = F.interpolate(
            weight,
            size=target_freq_len,
            mode="linear",
            align_corners=False
        )

        weight = weight.squeeze(0).transpose(0, 1)
        # [target_F, D]

        return weight

    def forward(self, x):
        """
        x: [B, N, D]
        """

        B, N, D = x.shape

        if D != self.d_model:
            raise ValueError(
                f"Input dim {D} does not match d_model={self.d_model}."
            )

        orig_dtype = x.dtype

        fft_input = (
            x.float()
            if x.dtype in [torch.float16, torch.bfloat16]
            else x
        )

        x_freq = torch.fft.rfft(fft_input, dim=1, norm="ortho")
        # [B, F, D]

        target_freq_len = x_freq.shape[1]

        real = self._resize_weight(self.weight_real, target_freq_len)
        imag = self._resize_weight(self.weight_imag, target_freq_len)

        real = real.to(device=x.device, dtype=fft_input.dtype)
        imag = imag.to(device=x.device, dtype=fft_input.dtype)

        complex_weight = torch.complex(real, imag)
        # [F, D]

        if self.use_filter_gate:
            gate = self._resize_weight(self.freq_gate, target_freq_len)
            gate = torch.sigmoid(gate).to(
                device=x.device,
                dtype=fft_input.dtype
            )
            complex_weight = complex_weight * gate

        x_freq = x_freq * complex_weight.unsqueeze(0)
        # [B, F, D]

        x_time = torch.fft.irfft(
            x_freq,
            n=N,
            dim=1,
            norm="ortho"
        )
        # [B, N, D]

        x_time = x_time.to(dtype=orig_dtype)

        out = self.proj(x_time)
        out = self.drop(out)

        return out


# =========================================================
# Token Self-Attention
# =========================================================
class TokenSelfAttention(nn.Module):
    """
    Standard self-attention over patch tokens.
    Used only as an attention replacement baseline for Filter1D.

    Input:
        x: [B, N, D]

    Output:
        out: [B, N, D]
    """

    def __init__(
        self,
        d_model,
        num_heads=4,
        dropout=0.1
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.attn(
            x,
            x,
            x,
            need_weights=False
        )

        out = self.drop(out)

        return out


# =========================================================
# Period-aware Patch Filter
# =========================================================
class PeriodAwarePatchFilter(nn.Module):
    """
    Multi-kernel period-aware patch filtering.

    It keeps the original multi-kernel depthwise convolution
    and adaptive filter gate.

    Input:
        x: [B, N, D]

    Output:
        out: [B, N, D]
    """

    def __init__(
        self,
        d_model,
        kernels=(3, 5, 7),
        dropout=0.1,
        filter_gate_temperature=1.0
    ):
        super().__init__()

        self.d_model = d_model
        self.kernels = tuple(kernels)
        self.filter_gate_temperature = float(filter_gate_temperature)

        if self.filter_gate_temperature <= 0:
            raise ValueError(
                "filter_gate_temperature must be positive, "
                f"got {self.filter_gate_temperature}."
            )

        for k in self.kernels:
            if k <= 0:
                raise ValueError(f"Kernel size must be positive, got {k}.")
            if k % 2 == 0:
                raise ValueError("All kernel sizes should be odd.")

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=k,
                padding=k // 2,
                groups=d_model,
                bias=True
            )
            for k in self.kernels
        ])

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(
                in_channels=d_model,
                out_channels=len(self.kernels),
                kernel_size=1,
                bias=True
            )
        )

        self.proj = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            bias=True
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, N, D]
        """

        x_t = x.transpose(1, 2)
        # [B, D, N]

        outs = torch.stack(
            [conv(x_t) for conv in self.convs],
            dim=1
        )
        # [B, K, D, N]

        gate = self.gate(x_t)
        # [B, K, 1]

        gate = (gate / self.filter_gate_temperature).softmax(
            dim=1
        ).unsqueeze(-1)
        # [B, K, 1, 1]

        out = (outs * gate).sum(dim=1)
        # [B, D, N]

        out = self.proj(out)
        out = self.drop(out)

        out = out.transpose(1, 2).contiguous()
        # [B, N, D]

        return out


# =========================================================
# Paimon Block
# =========================================================
class PaimonBlock(nn.Module):
    """
    Paimon block.

    Structure:
        PeriodAwarePatchFilter
        -> Filter1D / Attention
        -> FFN
    """

    def __init__(
        self,
        d_model,
        d_ff,
        seq_len,
        dropout=0.1,
        filter_kernels=(3, 5, 7),
        filter_gate_temperature=1.0,
        attention_replacement="filter1d",
        num_heads=4,
        filter_init_scale=0.02,
        use_filter_gate=True
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.attention_replacement = attention_replacement

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.patch_filter = PeriodAwarePatchFilter(
            d_model=d_model,
            kernels=filter_kernels,
            dropout=dropout,
            filter_gate_temperature=filter_gate_temperature
        )

        if attention_replacement == "filter1d":
            self.token_mixer = Filter1D(
                d_model=d_model,
                seq_len=seq_len,
                filter_init_scale=filter_init_scale,
                dropout=dropout,
                use_filter_gate=use_filter_gate
            )
        elif attention_replacement == "attention":
            self.token_mixer = TokenSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            raise ValueError(
                f"Unknown attention_replacement={attention_replacement}. "
                "Choose from ['filter1d', 'attention']."
            )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x: [B, N, D]
        """

        residual = x
        x = self.norm1(x)
        x = self.patch_filter(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.token_mixer(x)
        x = residual + x

        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + x

        return x


# =========================================================
# Paimon Encoder
# =========================================================
class PaimonEncoder(nn.Module):
    """
    Stacked Paimon blocks.
    """

    def __init__(
        self,
        depth,
        d_model,
        d_ff,
        seq_len,
        dropout=0.1,
        filter_kernels=(3, 5, 7),
        filter_gate_temperature=1.0,
        attention_replacement="filter1d",
        num_heads=4,
        filter_init_scale=0.02,
        use_filter_gate=True
    ):
        super().__init__()

        self.depth = depth

        self.blocks = nn.ModuleList([
            PaimonBlock(
                d_model=d_model,
                d_ff=d_ff,
                seq_len=seq_len,
                dropout=dropout,
                filter_kernels=filter_kernels,
                filter_gate_temperature=filter_gate_temperature,
                attention_replacement=attention_replacement,
                num_heads=num_heads,
                filter_init_scale=filter_init_scale,
                use_filter_gate=use_filter_gate
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        """
        x: [B, N, D]
        """

        for block in self.blocks:
            x = block(x)

        return x


# =========================================================
# Paimon
# =========================================================
class Paimon(nn.Module):
    """
    Paimon:
    Filtering Patch-Induced Redundancy for Time Series Forecasting.

    Input:
        x: [B, L, C]

    Output:
        y: [B, pred_len, C]
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        patch_len=16,
        stride=8,
        d_model=64,
        depth=2,
        d_ff=256,
        dropout=0.1,

        # Attention replacement.
        attention_replacement="filter1d",
        num_heads=4,

        # Patch-Aware Spectral Embedding.
        use_patch_spectral_embedding=True,
        pase_low_ratio=0.35,
        pase_mid_ratio=0.70,
        pase_gate_hidden_dim=None,
        pase_use_adaptive_gate=True,
        pase_use_learnable_freq_bias=True,
        pase_residual_scale_init=-1.0,
        pase_gate_temperature=1.0,
        pase_use_patch_norm=True,

        # Period-aware patch filter settings.
        period=24,
        period_rf_scale=1,

        # If filter_kernels is None / "auto":
        #   infer from period.
        # If filter_kernels is manually set:
        #   use manual kernels.
        filter_kernels=None,
        filter_kernel_target_num=3,
        filter_gate_temperature=1.0,

        # Filter1D settings.
        filter_init_scale=0.02,
        use_filter_gate=True,

        # Normalization.
        use_revin=True,
        revin_affine=True,
        revin_subtract_last=False,
        revin_eps=1e-5
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.depth = depth
        self.d_ff = d_ff

        self.attention_replacement = attention_replacement
        self.num_heads = num_heads

        if period is None:
            period = 24

        self.period = float(period)
        self.period_rf_scale = float(period_rf_scale)

        if self.period_rf_scale <= 0:
            raise ValueError(
                f"period_rf_scale must be positive, got {self.period_rf_scale}."
            )

        self.filter_kernel_target_num = int(filter_kernel_target_num)
        self.filter_gate_temperature = float(filter_gate_temperature)

        if self.filter_gate_temperature <= 0:
            raise ValueError(
                "filter_gate_temperature must be positive, "
                f"got {self.filter_gate_temperature}."
            )

        self.use_patch_spectral_embedding = use_patch_spectral_embedding
        self.use_revin = use_revin

        self.revin = RevIN(
            enc_in,
            eps=revin_eps,
            affine=revin_affine,
            subtract_last=revin_subtract_last
        ) if use_revin else None

        self.patch_num = (seq_len - patch_len) // stride + 1

        if self.patch_num <= 0:
            raise ValueError(
                f"Invalid patch setting: seq_len={seq_len}, "
                f"patch_len={patch_len}, stride={stride}."
            )

        # -------------------------------------------------
        # Infer filter_kernels from period if not specified.
        # -------------------------------------------------
        filter_kernels = _parse_kernels(filter_kernels)

        if filter_kernels is None:
            filter_kernels = infer_patch_filter_kernels(
                patch_num=self.patch_num,
                patch_len=patch_len,
                stride=stride,
                period=self.period,
                period_rf_scale=self.period_rf_scale,
                target_num=self.filter_kernel_target_num
            )

        self.filter_kernels = tuple(filter_kernels)

        if use_patch_spectral_embedding:
            self.patch = PatchAwareSpectralEmbedding(
                seq_len=seq_len,
                patch_len=patch_len,
                stride=stride,
                d_model=d_model,
                dropout=dropout,
                low_ratio=pase_low_ratio,
                mid_ratio=pase_mid_ratio,
                gate_hidden_dim=pase_gate_hidden_dim,
                use_adaptive_gate=pase_use_adaptive_gate,
                use_learnable_freq_bias=pase_use_learnable_freq_bias,
                residual_scale_init=pase_residual_scale_init,
                gate_temperature=pase_gate_temperature,
                use_patch_norm=pase_use_patch_norm
            )
        else:
            self.patch = StandardPatchEmbedding(
                patch_len=patch_len,
                stride=stride,
                d_model=d_model,
                dropout=dropout
            )

        self.encoder = PaimonEncoder(
            depth=depth,
            d_model=d_model,
            d_ff=d_ff,
            seq_len=self.patch_num,
            dropout=dropout,
            filter_kernels=self.filter_kernels,
            filter_gate_temperature=self.filter_gate_temperature,
            attention_replacement=attention_replacement,
            num_heads=num_heads,
            filter_init_scale=filter_init_scale,
            use_filter_gate=use_filter_gate
        )

        self.head = nn.Linear(
            self.patch_num * d_model,
            pred_len
        )

    def forward(self, x):
        """
        x: [B, L, C]
        """

        B, L, C = x.shape

        if L != self.seq_len:
            raise ValueError(
                f"Input sequence length {L} does not match seq_len={self.seq_len}."
            )

        if C != self.enc_in:
            raise ValueError(
                f"Input channel number {C} does not match enc_in={self.enc_in}."
            )

        if self.use_revin:
            x = self.revin(x, mode="norm")

        # Channel-independent forecasting.
        # [B, L, C] -> [B, C, L] -> [B*C, L]
        x_ci = x.permute(0, 2, 1).contiguous()
        x_ci = x_ci.reshape(B * C, L)
        # [B*C, L]

        tokens = self.patch(x_ci)
        # [B*C, N, D]

        z = self.encoder(tokens)
        # [B*C, N, D]

        z = z.reshape(B * C, -1)
        # [B*C, N*D]

        out = self.head(z)
        # [B*C, pred_len]

        y = out.reshape(B, C, self.pred_len)
        y = y.permute(0, 2, 1).contiguous()
        # [B, pred_len, C]

        if self.use_revin:
            y = self.revin(y, mode="denorm")

        return y