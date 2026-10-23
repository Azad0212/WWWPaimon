def list_to_str(x):
    if x is None:
        return "None"
    if isinstance(x, (list, tuple)):
        return "-".join(map(str, x))
    return str(x)


def build_best_model_name(args):
    model_name = args.model_name.lower()

    base = [
        model_name,
        args.data_type,
        f"sl{args.seq_len}",
        f"ll{args.label_len}",
        f"pl{args.pred_len}",
    ]


    if model_name == "logot":
        extra = [
            f"patch{args.patch_len}",
            f"pstride{args.patch_stride}",
            f"dstride{args.data_stride}",
            f"period{args.period}",
            f"dim{args.d_model}",
            f"depth{list_to_str(args.depths)}",
            f"heads{list_to_str(args.num_heads)}",
            f"mlp{args.mlp_ratio}",
            f"pool{args.pooling}",
            f"drop{args.dropout}",
            f"revin{args.revin}",
            f"aff{args.affine}",
            f"sublast{args.subtract_last}",
        ]
    else:
        extra = [
            f"patch{args.patch_len}",
            f"pstride{args.patch_stride}",
            f"dim{args.d_model}",
            f"drop{args.dropout}",
        ]

    return "_".join(base + extra) + "_best.pth"
