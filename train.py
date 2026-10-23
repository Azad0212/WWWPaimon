import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
import argparse
import os

from utils.data_factory import load_data
from utils.timefeatures import infer_time_dim
from utils.logot_configs import process_logot_args
from utils.best_name import build_best_model_name
from model.get_model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Paimon Time-Series Forecasting")


    # ============================================================
    parser.add_argument('--data_type', type=str, default='etth1')
    parser.add_argument('--root_path', type=str, default='./data')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--time_col', type=str, default='date')
    parser.add_argument('--model_name', type=str, default='paimon')
    parser.add_argument('--timeenc', type=int, default=1)
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--scale', type=int, default=1)

    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--patch_stride', type=int, default=8)

    parser.add_argument('--data_stride', type=int, default=1)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--save_dir', type=str, default='result_forecast')


    # ============================================================
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--enc_layers', type=int, default=3)       # fallback for depth
    parser.add_argument('--depth', type=int, default=None,      
                        help='Number of Paimon encoder layers (overrides enc_layers if set)')



    # ============================================================
    # ---- Patch-Aware Spectral Embedding ----
    parser.add_argument('--use_patch_spectral_embedding', type=int, default=1)
    parser.add_argument('--pase_low_ratio', type=float, default=0.35)
    parser.add_argument('--pase_mid_ratio', type=float, default=0.70)
    parser.add_argument('--pase_gate_hidden_dim', type=int, default=None)
    parser.add_argument('--pase_use_adaptive_gate', type=int, default=1)
    parser.add_argument('--pase_use_learnable_freq_bias', type=int, default=1)
    parser.add_argument('--pase_residual_scale_init', type=float, default=-1.0)
    parser.add_argument('--pase_gate_temperature', type=float, default=1.0)
    parser.add_argument('--pase_use_patch_norm', type=int, default=1)

    # ---- Period-aware Patch Filter ----
    parser.add_argument('--period', type=int, default=24)                    
    parser.add_argument('--period_rf_scale', type=float, default=1.0)
    parser.add_argument('--filter_kernels', type=str, default=None)
    parser.add_argument('--filter_kernel_target_num', type=int, default=3)
    parser.add_argument('--filter_gate_temperature', type=float, default=1.0)

    # ---- Attention replacement ----
    parser.add_argument('--attention_replacement', type=str, default='filter1d',
                        choices=['filter1d', 'attention'])
    parser.add_argument('--filter_init_scale', type=float, default=0.02)
    parser.add_argument('--use_filter_gate', type=int, default=1)

    # ---- RevIN ----
    parser.add_argument('--use_revin', type=int, default=1)
    parser.add_argument('--affine', type=int, default=1)
    parser.add_argument('--subtract_last', type=int, default=0)
    parser.add_argument('--revin_eps', type=float, default=1e-5)

    return parser.parse_args()


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inverse_transform_3d(arr_3d: np.ndarray, scaler):
    """
    arr_3d: (B, H, C)
    scaler: sklearn StandardScaler
    """
    if scaler is None:
        return arr_3d

    B, H, C = arr_3d.shape
    arr_2d = arr_3d.reshape(-1, C)
    arr_2d = scaler.inverse_transform(arr_2d)
    return arr_2d.reshape(B, H, C)


def compute_metrics(preds: np.ndarray, gts: np.ndarray):
    """
    preds, gts: (N, H, C), already in original scale
    """
    eps = 1e-8

    diff = preds - gts
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(mse)

    numerator = np.abs(preds - gts)
    denominator = (np.abs(gts) + np.abs(preds)) / 2.0
    smape = np.mean(numerator / (denominator + eps))

    wape = np.sum(np.abs(preds - gts)) / (np.sum(np.abs(gts)) + eps)

    rmse_global = np.sqrt(np.mean(diff ** 2))
    scale_true = np.sqrt(np.mean(gts ** 2))
    scale_pred = np.sqrt(np.mean(preds ** 2))
    tic = rmse_global / (scale_true + scale_pred + eps)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "wape": wape,
        "tic": tic,
    }


def align_to_pred_len(tensor, pred_len):
    """
    Unified alignment:
    keep only the last pred_len on time dimension.
    tensor: [B, T, ...] or None
    """
    if tensor is None:
        return None
    return tensor[:, -pred_len:, ...]


def build_decoder_input(y, label_len, pred_len, device):
    """
    y: [B, label_len + pred_len, C]
    dec_inp: [B, label_len + pred_len, C]
        = [known history tail(label_len), zeros(pred_len)]
    """
    if y is None:
        return None

    if y.size(1) < label_len + pred_len:
        raise ValueError(
            f"y length {y.size(1)} is smaller than label_len + pred_len = {label_len + pred_len}"
        )

    dec_zeros = torch.zeros_like(y[:, -pred_len:, :]).float()
    dec_inp = torch.cat([y[:, :label_len, :], dec_zeros], dim=1).float().to(device)
    return dec_inp


def is_decoder_model(model_name: str):
    """
    Models in this list will use:
        model(x, x_mark, dec_inp, y_mark)
    Modify this list according to your actual decoder-based models.
    """
    decoder_models = {
        "transformer",
        "autoformer",
        "informer",
        "fedformer",
        "itransformer",
        # "gcvit",
    }
    return model_name.lower() in decoder_models


# ============================================================
# Train
# ============================================================

def train(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    # args = process_logot_args(args)
    # --------------------------------------------------------
    # 1) Load data
    # --------------------------------------------------------
    train_loader, val_loader, test_loader, input_dim, scaler = load_data(args)

    print(f"[INFO] Dataset={args.data_type}, input_dim={input_dim}")
    print(f"[INFO] Device={device}")

    # --------------------------------------------------------
    # 2) Build model
    # --------------------------------------------------------
    args.time_dim = infer_time_dim(args.freq, args.timeenc)
    model = get_model(args, input_dim=input_dim).to(device)
    print("Model device:", next(model.parameters()).device)

    # --------------------------------------------------------
    # 3) Loss & Optimizer
    # --------------------------------------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --------------------------------------------------------
    # 4) Paths
    # --------------------------------------------------------
    best_model_path = os.path.join(args.save_dir, build_best_model_name(args))

    # --------------------------------------------------------
    # 5) Training loop
    # --------------------------------------------------------
    best_val_loss = float("inf")
    early_stop_cnt = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for idx, batch in enumerate(train_loader):
            x, y, x_mark, y_mark, cycle = batch
            x_mark = x_mark.to(device)
            y_mark = y_mark.to(device)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            if is_decoder_model(args.model_name):
                dec_inp = build_decoder_input(y, args.label_len, args.pred_len, device)
                output = model(x, x_mark, dec_inp, y_mark)
            elif args.model_name.lower() == 'tqnet':
                cycle = cycle.to(device)
                output = model(x, cycle)
            else:
                output = model(x)

            output = align_to_pred_len(output, args.pred_len)
            y = align_to_pred_len(y, args.pred_len)

            if output.shape != y.shape:
                raise ValueError(
                    f"[TRAIN] Model output shape {output.shape} does not match target shape {y.shape}"
                )

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if idx % 50 == 0:
                print(f"Epoch {epoch + 1} | idx {idx} | Train MSE {loss.item():.6f}")

        epoch_loss /= max(len(train_loader), 1)
        print(f"[TRAIN] Epoch {epoch + 1}, MSE={epoch_loss:.6f}, time={time.time() - t0:.1f}s")

        # ----------------------------------------------------
        # 6) Validation
        # ----------------------------------------------------
        model.eval()

        val_loss = 0.0
        val_start_time = time.time()

        with torch.no_grad():
            for batch in val_loader:
                x, y, x_mark, y_mark, cycle = batch
                x_mark = x_mark.to(device)
                y_mark = y_mark.to(device)
                x = x.to(device)
                y = y.to(device)

                if is_decoder_model(args.model_name):
                    dec_inp = build_decoder_input(y, args.label_len, args.pred_len, device)
                    output = model(x, x_mark, dec_inp, y_mark)
                elif args.model_name.lower() == 'tqnet':
                    cycle = cycle.to(device)
                    output = model(x, cycle)
                else:
                    output = model(x)

                output = align_to_pred_len(output, args.pred_len)
                y = align_to_pred_len(y, args.pred_len)

                if output.shape != y.shape:
                    raise ValueError(
                        f"[VAL] Model output shape {output.shape} does not match target shape {y.shape}"
                    )

                loss = criterion(output, y)
                val_loss += loss.item()

        val_time = time.time() - val_start_time
        val_loss /= max(len(val_loader), 1)

        print(f"[VAL] Epoch {epoch + 1}, MSE={val_loss:.6f}, Time={val_time:.4f}s")
        # ----------------------------------------------------
        # 7) Early stopping
        # ----------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_cnt = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] Best model saved to: {best_model_path}")
        else:
            early_stop_cnt += 1
            print(f"[INFO] Early stop counter: {early_stop_cnt}/{args.patience}")
            if early_stop_cnt >= args.patience:
                print(f"[Early Stop] triggered at epoch {epoch + 1}")
                break

    # --------------------------------------------------------
    # 8) Test
    # --------------------------------------------------------
    print("[INFO] Loading best model for test...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    preds = []
    gts = []

    with torch.no_grad():
        for batch in test_loader:
            x, y, x_mark, y_mark, cycle = batch
            x_mark = x_mark.to(device)
            y_mark = y_mark.to(device)
            x = x.to(device)
            y = y.to(device)

            if is_decoder_model(args.model_name):
                dec_inp = build_decoder_input(y, args.label_len, args.pred_len, device)
                output = model(x, x_mark, dec_inp, y_mark)
            elif args.model_name.lower() == 'tqnet':
                cycle = cycle.to(device)
                output = model(x, cycle)
            else:
                output = model(x)

            output = align_to_pred_len(output, args.pred_len)
            y = align_to_pred_len(y, args.pred_len)

            if output.shape != y.shape:
                raise ValueError(
                    f"[TEST] Model output shape {output.shape} does not match target shape {y.shape}"
                )

            preds.append(output.detach().cpu().numpy())
            gts.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)

    # preds_inv = inverse_transform_3d(preds, scaler)
    # gts_inv = inverse_transform_3d(gts, scaler)
    # metrics = compute_metrics(preds_inv, gts_inv)

    metrics = compute_metrics(preds, gts)

    print(
        f"[TEST] "
        f"Model={args.model_name} | "
        f"Data={args.data_type} | "
        f"Prediction={args.pred_len} | "
        f"MSE={metrics['mse']:.4f} | "
        f"MAE={metrics['mae']:.4f} | "
        f"RMSE={metrics['rmse']:.4f}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = parse_args()
    train(args)
