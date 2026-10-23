#PatchTST Xpatch ParallelTime TQNet DLinear CATS DeformableTST
import math
import torch
import torch.nn as nn
# ============================================================

# ============================================================

# model/get_model.py


from .Paimon import Paimon


def get_model(args, input_dim):
    name = args.model_name.lower()

    if name in ["paimon", "pai"]:
        depth = getattr(args, "depth", None)
        if depth is None:
            depth = getattr(args, "enc_layers", 3)

        return Paimon(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=input_dim,

            patch_len=args.patch_len,
            stride=args.patch_stride,

            d_model=args.d_model,
            depth=depth,                  
            d_ff=args.d_ff,
            dropout=args.dropout,

            # Patch-Aware Spectral Embedding
            use_patch_spectral_embedding=bool(args.use_patch_spectral_embedding),
            pase_low_ratio=args.pase_low_ratio,
            pase_mid_ratio=args.pase_mid_ratio,
            pase_gate_hidden_dim=args.pase_gate_hidden_dim,
            pase_use_adaptive_gate=bool(args.pase_use_adaptive_gate),
            pase_use_learnable_freq_bias=bool(args.pase_use_learnable_freq_bias),
            pase_residual_scale_init=args.pase_residual_scale_init,
            pase_gate_temperature=args.pase_gate_temperature,
            pase_use_patch_norm=bool(args.pase_use_patch_norm),

            # Period-aware Patch Filter
            period=args.period,
            period_rf_scale=args.period_rf_scale,
            filter_kernels=args.filter_kernels,
            filter_kernel_target_num=args.filter_kernel_target_num,
            filter_gate_temperature=args.filter_gate_temperature,

            # Attention replacement
            attention_replacement=args.attention_replacement,
            filter_init_scale=args.filter_init_scale,
            use_filter_gate=bool(args.use_filter_gate),

            # RevIN
            use_revin=bool(args.use_revin),
            revin_affine=bool(args.affine),
            revin_subtract_last=bool(args.subtract_last),
            revin_eps=args.revin_eps,
        )

    raise ValueError(f"Unsupported model: {args.model_name}. Only 'paimon' is allowed.")
