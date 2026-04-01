#!/usr/bin/env python3
"""
CaLaM risk/feature training script v6.
Trains:
1. DynamicContextFeatureExtractor.W_risk -> w_risk_v2.pt
2. RiskModel context head -> risk_model.pt

Key hardening:
- group-aware train/val/test split by entry_id to prevent RTP sample leakage
- shared feature scoring path between training and inference
- fail-fast checks for non-finite tensors and legacy datasets without grouping metadata
"""

# P0: real-time logs
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.stdout.reconfigure(line_buffering=True)

def _preset_cuda_visible_devices_from_argv(argv):
    """Honor --gpu before importing torch so CUDA device masking is deterministic."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    for idx, arg in enumerate(argv):
        if arg == "--gpu" and idx + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[idx + 1]
            return
        if arg.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.split("=", 1)[1]
            return

_preset_cuda_visible_devices_from_argv(sys.argv[1:])

import argparse
import logging
import random

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import DynamicContextFeatureExtractor
from src.risk import RiskModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

def build_group_masks(entry_ids: torch.Tensor, train_split: float, val_split: float, seed: int):
    if entry_ids.dim() != 1:
        raise ValueError(f"entry_ids must be rank-1, got shape {tuple(entry_ids.shape)}")

    unique_entry_ids = sorted({int(x) for x in entry_ids.tolist()})
    num_groups = len(unique_entry_ids)
    if num_groups < 3:
        raise ValueError(
            f"Need at least 3 unique entry_ids for grouped split, found {num_groups}."
        )

    rng = random.Random(seed)
    rng.shuffle(unique_entry_ids)

    train_groups = max(1, int(num_groups * train_split))
    val_groups = max(1, int(num_groups * val_split))
    if train_groups + val_groups >= num_groups:
        val_groups = max(1, num_groups - train_groups - 1)
    test_groups = num_groups - train_groups - val_groups
    if test_groups < 1:
        raise ValueError(
            f"Grouped split leaves no test groups: train={train_groups}, val={val_groups}, total={num_groups}"
        )

    train_group_ids = set(unique_entry_ids[:train_groups])
    val_group_ids = set(unique_entry_ids[train_groups:train_groups + val_groups])
    test_group_ids = set(unique_entry_ids[train_groups + val_groups:])

    entry_id_list = [int(x) for x in entry_ids.tolist()]
    train_mask = torch.tensor([eid in train_group_ids for eid in entry_id_list], dtype=torch.bool)
    val_mask = torch.tensor([eid in val_group_ids for eid in entry_id_list], dtype=torch.bool)
    test_mask = torch.tensor([eid in test_group_ids for eid in entry_id_list], dtype=torch.bool)

    if train_mask.sum().item() == 0 or val_mask.sum().item() == 0 or test_mask.sum().item() == 0:
        raise RuntimeError(
            "Grouped split produced an empty split. "
            f"train={train_mask.sum().item()} val={val_mask.sum().item()} test={test_mask.sum().item()}"
        )

    return train_mask, val_mask, test_mask, {
        "num_entry_groups": num_groups,
        "train_entry_groups": len(train_group_ids),
        "val_entry_groups": len(val_group_ids),
        "test_entry_groups": len(test_group_ids),
    }

def compute_binary_metrics(preds: torch.Tensor, labels: torch.Tensor, threshold: float):
    binary_preds = (preds > threshold).float()
    binary_labels = (labels > threshold).float()
    accuracy = (binary_preds == binary_labels).float().mean().item()

    toxic_mask = binary_labels == 1.0
    recall = (binary_preds[toxic_mask] == 1.0).float().mean().item() if toxic_mask.any() else 0.0

    pred_toxic_mask = binary_preds == 1.0
    precision = (
        (binary_labels[pred_toxic_mask] == 1.0).float().mean().item()
        if pred_toxic_mask.any() else 0.0
    )
    return accuracy, recall, precision

def evaluate(feature_model, risk_model, loader, device, toxic_threshold):
    feature_model.eval()
    risk_model.eval()

    feature_loss_sum = 0.0
    risk_loss_sum = 0.0
    total = 0
    feature_preds = []
    risk_preds = []
    labels = []

    with torch.no_grad():
        for h, token_ids, score in loader:
            h = h.to(device)
            token_ids = token_ids.to(device)
            score = score.to(device)

            feature_pred = feature_model.score_tokens(h, token_ids)
            risk_pred = risk_model(h)

            feature_loss = F.mse_loss(feature_pred, score)
            risk_loss = F.mse_loss(risk_pred, score)

            batch_size = len(h)
            feature_loss_sum += feature_loss.item() * batch_size
            risk_loss_sum += risk_loss.item() * batch_size
            total += batch_size

            feature_preds.append(feature_pred.cpu())
            risk_preds.append(risk_pred.cpu())
            labels.append(score.cpu())

    feature_preds_t = torch.cat(feature_preds)
    risk_preds_t = torch.cat(risk_preds)
    labels_t = torch.cat(labels)

    feature_mse = feature_loss_sum / total
    risk_mse = risk_loss_sum / total
    risk_accuracy, risk_recall, risk_precision = compute_binary_metrics(
        risk_preds_t, labels_t, toxic_threshold
    )

    return {
        "feature_mse": feature_mse,
        "risk_mse": risk_mse,
        "risk_accuracy": risk_accuracy,
        "risk_recall": risk_recall,
        "risk_precision": risk_precision,
        "feature_preds": feature_preds_t,
        "risk_preds": risk_preds_t,
        "labels": labels_t,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/jhdx0003008/data/calam/checkpoints/risk_train_data.pt")
    parser.add_argument("--output", default="/jhdx0003008/data/calam/checkpoints/w_risk_v2.pt")
    parser.add_argument("--risk_output", default="/jhdx0003008/data/calam/checkpoints/risk_model.pt")
    parser.add_argument("--embedding_model", default="/jhdx0003008/models/Qwen2.5-14B-Instruct")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--toxic_threshold", type=float, default=0.3,
                        help="Binary threshold for toxicity-oriented recall checks")
    parser.add_argument("--train_split", type=float, default=0.70)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for CUDA_VISIBLE_DEVICES")
    parser.add_argument("--run_name", type=str, default="risk_model_training")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--wandb_project", default="calam")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context_loss_weight", type=float, default=1.0,
                        help="Weight applied to the standalone RiskModel loss")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Global grad-norm clip to prevent W_risk/RiskModel overflow")
    parser.add_argument("--feature_logit_clamp", type=float, default=30.0,
                        help="Clamp applied before feature sigmoid to avoid saturation overflow")
    parser.add_argument("--feature_projection_clamp", type=float, default=256.0,
                        help="Clamp applied to W_risk(h_t) before token bilinear scoring")
    parser.add_argument("--risk_input_clamp", type=float, default=256.0,
                        help="Clamp applied to hidden states before the RiskModel MLP")
    parser.add_argument("--allow_legacy_ungrouped_split", action="store_true",
                        help="Allow training on old datasets that do not contain entry_ids")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output_path = os.path.abspath(args.output)
    risk_output_path = os.path.abspath(args.risk_output)
    if output_path == risk_output_path:
        raise ValueError(
            "--output and --risk_output resolve to the same file. "
            "Refusing to overwrite W_risk and RiskModel checkpoints with one path."
        )
    if args.max_grad_norm <= 0.0:
        raise ValueError(f"--max_grad_norm must be > 0, got {args.max_grad_norm}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(risk_output_path) or ".", exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # === Load pretrained embeddings for dynamic probing ===
    logger.info(f"Loading language model for embedding matrix extraction: {args.embedding_model}")
    llm = AutoModelForCausalLM.from_pretrained(
        args.embedding_model,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    embedding_matrix = llm.get_input_embeddings().weight.detach().float()
    logger.info(f"Loaded embedding matrix: {embedding_matrix.shape}")
    del llm

    # === W&B init ===
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config={
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "train_split": args.train_split,
                    "val_split": args.val_split,
                    "gpu": args.gpu,
                    "weight_decay": args.weight_decay,
                    "patience": args.patience,
                    "warmup_epochs": args.warmup_epochs,
                    "toxic_threshold": args.toxic_threshold,
                    "scheduler": "ReduceLROnPlateau(factor=0.5, patience=3)",
                    "feature_architecture": "h_t -> W_risk -> <E_v> -> sigmoid",
                    "risk_architecture": "MLP(hidden->512->64->1)->sigmoid",
                    "task": "joint regression (feature + context risk)",
                    "resumed": args.resume,
                    "context_loss_weight": args.context_loss_weight,
                    "max_grad_norm": args.max_grad_norm,
                    "feature_logit_clamp": args.feature_logit_clamp,
                    "feature_projection_clamp": args.feature_projection_clamp,
                    "risk_input_clamp": args.risk_input_clamp,
                },
                tags=["risk_model", "training", "group_split"],
            )
            logger.info("W&B initialized")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}, continuing without W&B")
            args.use_wandb = False

    # === Load data ===
    logger.info(f"Loading data from {args.data}...")
    data = torch.load(args.data, map_location="cpu")

    hidden_states = data["hidden_states"].float()
    toxicity_scores = data["toxicity_scores"].float()
    next_tokens = data.get("next_tokens")
    entry_ids = data.get("entry_ids")
    hidden_size = int(data["hidden_size"])

    if next_tokens is None:
        raise RuntimeError("[FAIL-FAST] Dataset missing next_tokens. Please re-run collect_risk_data.py")
    next_tokens = next_tokens.long()

    if hidden_states.dim() != 2:
        raise ValueError(f"hidden_states must have shape (N, H), got {tuple(hidden_states.shape)}")
    if toxicity_scores.dim() != 1 or next_tokens.dim() != 1:
        raise ValueError(
            f"Expected toxicity_scores and next_tokens to be rank-1, got "
            f"{tuple(toxicity_scores.shape)} and {tuple(next_tokens.shape)}"
        )
    if hidden_states.shape[0] != toxicity_scores.shape[0] or hidden_states.shape[0] != next_tokens.shape[0]:
        raise ValueError(
            f"Dataset cardinality mismatch: hidden={hidden_states.shape[0]} "
            f"scores={toxicity_scores.shape[0]} next_tokens={next_tokens.shape[0]}"
        )
    if hidden_states.shape[1] != hidden_size:
        raise ValueError(
            f"hidden_size metadata mismatch: tensor={hidden_states.shape[1]} metadata={hidden_size}"
        )
    if not torch.isfinite(hidden_states).all():
        raise FloatingPointError("Training data contains non-finite hidden_states.")
    if not torch.isfinite(toxicity_scores).all():
        raise FloatingPointError("Training data contains non-finite toxicity_scores.")
    if toxicity_scores.min().item() < 0.0 or toxicity_scores.max().item() > 1.0:
        raise ValueError(
            f"Toxicity scores must stay in [0, 1], got min={toxicity_scores.min().item()} "
            f"max={toxicity_scores.max().item()}"
        )

    if entry_ids is None:
        if not args.allow_legacy_ungrouped_split:
            raise RuntimeError(
                "[FAIL-FAST] Dataset missing entry_ids. Re-run collect_risk_data.py with the hardened collector "
                "to avoid train/val/test leakage across prompt positions."
            )
        logger.warning("Legacy dataset without entry_ids detected; falling back to ungrouped split.")
        entry_ids = torch.arange(hidden_states.shape[0], dtype=torch.long)
    else:
        entry_ids = entry_ids.long()
        if entry_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                f"entry_ids cardinality mismatch: entry_ids={entry_ids.shape[0]} samples={hidden_states.shape[0]}"
            )

    logger.info(
        f"Data: {hidden_states.shape[0]} samples, hidden_size={hidden_size}, "
        f"unique_entries={entry_ids.unique().numel()}"
    )
    logger.info(
        f"Toxicity stats: mean={toxicity_scores.mean():.3f}, std={toxicity_scores.std():.3f}, "
        f"toxic(>0.5)={(toxicity_scores > 0.5).sum().item()} "
        f"({(toxicity_scores > 0.5).float().mean() * 100:.1f}%)"
    )

    if args.use_wandb:
        import wandb
        wandb.config.update({
            "num_samples": hidden_states.shape[0],
            "hidden_size": hidden_size,
            "toxic_ratio": (toxicity_scores > 0.5).float().mean().item(),
            "unique_entries": entry_ids.unique().numel(),
        })

    # === Grouped train/val/test split ===
    train_mask, val_mask, test_mask, split_info = build_group_masks(
        entry_ids=entry_ids,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
    )

    train_dataset = TensorDataset(
        hidden_states[train_mask], next_tokens[train_mask], toxicity_scores[train_mask]
    )
    val_dataset = TensorDataset(
        hidden_states[val_mask], next_tokens[val_mask], toxicity_scores[val_mask]
    )
    test_dataset = TensorDataset(
        hidden_states[test_mask], next_tokens[test_mask], toxicity_scores[test_mask]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    logger.info(
        f"Grouped split: train={train_size}, val={val_size}, test={test_size} | "
        f"entry groups train/val/test={split_info['train_entry_groups']}/"
        f"{split_info['val_entry_groups']}/{split_info['test_entry_groups']}"
    )

    # === Models + optimizer ===
    feature_model = DynamicContextFeatureExtractor(
        hidden_size=hidden_size,
        embedding_matrix=embedding_matrix.to(args.device),
        device=args.device,
        logit_clamp=args.feature_logit_clamp,
        projection_clamp=args.feature_projection_clamp,
    )
    risk_model = RiskModel(
        hidden_size=hidden_size,
        dropout=args.dropout,
        input_clamp=args.risk_input_clamp,
    ).to(args.device)
    trainable_params = list(feature_model.parameters()) + list(risk_model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    start_epoch = 0

    full_ckpt_path = args.output.replace(".pt", "_full.pt")
    if args.resume:
        if os.path.exists(full_ckpt_path):
            ckpt = torch.load(full_ckpt_path, map_location=args.device)
            if "feature_model_state" in ckpt:
                feature_model.load_state_dict(ckpt["feature_model_state"])
                risk_model.load_state_dict(ckpt["risk_model_state"])
            elif "model_state" in ckpt:
                feature_model.load_state_dict(ckpt["model_state"])
                logger.warning("Resumed feature_model from legacy checkpoint without risk_model state.")
            else:
                raise RuntimeError(f"Unsupported checkpoint schema in {full_ckpt_path}")
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = int(ckpt.get("epoch", 0))
            logger.info(f"Resumed from {full_ckpt_path} (epoch {start_epoch})")
        else:
            if os.path.exists(args.output):
                feature_model.load_state_dict(torch.load(args.output, map_location=args.device))
                logger.info(f"Loaded feature_model weights from {args.output}")
            if os.path.exists(args.risk_output):
                risk_model.load_state_dict(torch.load(args.risk_output, map_location=args.device))
                logger.info(f"Loaded risk_model weights from {args.risk_output}")

    logger.info(
        f"Feature params: {sum(p.numel() for p in feature_model.parameters()):,} | "
        f"Risk params: {sum(p.numel() for p in risk_model.parameters()):,}"
    )

    # === Training with early stopping ===
    best_val_loss = float("inf")
    best_feature_state = None
    best_risk_state = None
    best_epoch = 0
    best_val_feature_mse = None
    best_val_risk_mse = None
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        feature_model.train()
        risk_model.train()
        train_feature_loss_sum = 0.0
        train_risk_loss_sum = 0.0
        epoch_max_grad_norm = 0.0

        for h, token_ids, score in train_loader:
            h = h.to(args.device)
            token_ids = token_ids.to(args.device)
            score = score.to(args.device)

            feature_pred = feature_model.score_tokens(h, token_ids)
            risk_pred = risk_model(h)

            feature_loss = F.mse_loss(feature_pred, score)
            risk_loss = F.mse_loss(risk_pred, score)
            loss = feature_loss + args.context_loss_weight * risk_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch + 1}: "
                    f"feature_loss={feature_loss.item()} risk_loss={risk_loss.item()}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=args.max_grad_norm,
                error_if_nonfinite=True,
            )
            grad_norm_value = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            epoch_max_grad_norm = max(epoch_max_grad_norm, grad_norm_value)
            optimizer.step()

            batch_size = len(h)
            train_feature_loss_sum += feature_loss.item() * batch_size
            train_risk_loss_sum += risk_loss.item() * batch_size

        train_feature_mse = train_feature_loss_sum / train_size
        train_risk_mse = train_risk_loss_sum / train_size

        val_metrics = evaluate(
            feature_model=feature_model,
            risk_model=risk_model,
            loader=val_loader,
            device=args.device,
            toxic_threshold=args.toxic_threshold,
        )
        val_total_loss = val_metrics["feature_mse"] + args.context_loss_weight * val_metrics["risk_mse"]

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch + 1:2d}/{args.epochs}: "
            f"train_feature_mse={train_feature_mse:.4f} train_risk_mse={train_risk_mse:.4f} "
            f"val_feature_mse={val_metrics['feature_mse']:.4f} val_risk_mse={val_metrics['risk_mse']:.4f} "
            f"risk_acc={val_metrics['risk_accuracy']:.3f} risk_recall={val_metrics['risk_recall']:.3f} "
            f"risk_prec={val_metrics['risk_precision']:.3f} lr={current_lr:.1e} "
            f"max_grad_norm={epoch_max_grad_norm:.3f}"
        )

        if args.use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train/feature_mse": train_feature_mse,
                "train/risk_mse": train_risk_mse,
                "val/feature_mse": val_metrics["feature_mse"],
                "val/risk_mse": val_metrics["risk_mse"],
                "val/risk_accuracy": val_metrics["risk_accuracy"],
                "val/risk_recall": val_metrics["risk_recall"],
                "val/risk_precision": val_metrics["risk_precision"],
                "lr": current_lr,
                "train/max_grad_norm": epoch_max_grad_norm,
            })

        if epoch >= args.warmup_epochs:
            scheduler.step(val_total_loss)

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_feature_state = {k: v.cpu().clone() for k, v in feature_model.state_dict().items()}
            best_risk_state = {k: v.cpu().clone() for k, v in risk_model.state_dict().items()}
            best_epoch = epoch + 1
            best_val_feature_mse = val_metrics["feature_mse"]
            best_val_risk_mse = val_metrics["risk_mse"]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch + 1} (patience={args.patience})")
                break

    # === Save checkpoints ===
    if best_feature_state is None or best_risk_state is None:
        raise RuntimeError("Training finished without a best checkpoint.")

    full_ckpt = {
        "feature_model_state": best_feature_state,
        "risk_model_state": best_risk_state,
        "optimizer_state": optimizer.state_dict(),
        "epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_feature_mse": best_val_feature_mse,
        "best_val_risk_mse": best_val_risk_mse,
    }
    torch.save(full_ckpt, full_ckpt_path)
    torch.save(best_feature_state, args.output)
    torch.save(best_risk_state, args.risk_output)
    logger.info(
        f"Saved feature checkpoint to {args.output} and risk checkpoint to {args.risk_output} "
        f"(best_epoch={best_epoch}, joint_val_loss={best_val_loss:.4f})"
    )

    # === Test evaluation ===
    feature_model.load_state_dict(best_feature_state)
    risk_model.load_state_dict(best_risk_state)
    feature_model.to(args.device)
    risk_model.to(args.device)

    test_metrics = evaluate(
        feature_model=feature_model,
        risk_model=risk_model,
        loader=test_loader,
        device=args.device,
        toxic_threshold=args.toxic_threshold,
    )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"TEST SET RESULTS (held-out, {test_size} samples)")
    logger.info(f"   feature_test_mse={test_metrics['feature_mse']:.4f}")
    logger.info(f"   risk_test_mse={test_metrics['risk_mse']:.4f}")
    logger.info(f"   risk_test_accuracy={test_metrics['risk_accuracy']:.3f}")
    logger.info(f"   risk_test_recall={test_metrics['risk_recall']:.3f}")
    logger.info(f"   risk_test_precision={test_metrics['risk_precision']:.3f}")
    logger.info(f"{'=' * 60}")

    checks_passed = 0
    total_checks = 2
    if test_metrics["risk_recall"] > 0.75:
        logger.info(f"[1/2] risk_test_recall={test_metrics['risk_recall']:.3f} > 0.75 - PASS")
        checks_passed += 1
    else:
        logger.warning(f"[1/2] risk_test_recall={test_metrics['risk_recall']:.3f} <= 0.75 - NEEDS TUNING")

    if test_metrics["risk_mse"] < 0.06:
        logger.info(f"[2/2] risk_test_mse={test_metrics['risk_mse']:.4f} < 0.06 - PASS")
        checks_passed += 1
    else:
        logger.warning(f"[2/2] risk_test_mse={test_metrics['risk_mse']:.4f} >= 0.06 - NEEDS TUNING")

    logger.info(f"RiskModel: {checks_passed}/{total_checks} checks passed")

    if args.use_wandb:
        import wandb
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["feature_test_mse"] = test_metrics["feature_mse"]
        wandb.summary["risk_test_mse"] = test_metrics["risk_mse"]
        wandb.summary["risk_test_accuracy"] = test_metrics["risk_accuracy"]
        wandb.summary["risk_test_recall"] = test_metrics["risk_recall"]
        wandb.summary["risk_test_precision"] = test_metrics["risk_precision"]
        wandb.summary["checks_passed"] = f"{checks_passed}/{total_checks}"
        wandb.finish()

if __name__ == "__main__":
    main()
