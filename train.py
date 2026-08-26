import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader, DynamicBatchSampler

from helpers import finish_optimizer_step, seed_everything, training_step
from model import FlowEGNN, load_graph_dataset, load_model_checkpoint


CHECKPOINT_FORMAT_VERSION = 2


def infer_stats_path(data_path):
    """Infer the generated statistics file from a training dataset path."""
    data_path = Path(data_path)
    name = data_path.name
    if name.endswith("_data_norm.pt"):
        stem = name.removesuffix("_data_norm.pt")
    elif name.endswith("_data_norm_on_disk"):
        stem = name.removesuffix("_data_norm_on_disk")
    else:
        return None
    candidate = data_path.parent / f"{stem}_stats.pt"
    return candidate if candidate.exists() else None


def load_normalization_stats(data_path, stats_path=None):
    if stats_path is None:
        stats_path = infer_stats_path(data_path)
        if stats_path is None:
            raise FileNotFoundError(
                "Could not infer the normalization-statistics file from "
                f"{data_path!s}; pass stats_path or --stats explicitly"
            )
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization statistics not found: {stats_path}")
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    return stats, stats_path


def portable_source_name(path):
    """Record artifact provenance without embedding a machine-specific path."""
    return Path(path).name


def make_checkpoint(
    *,
    epoch,
    model,
    optimiser,
    scaler,
    model_config,
    training_config,
    normalization_stats,
    stats_path,
    data_path,
):
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimiser.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "model_config": model_config,
        "training_config": training_config,
        "normalization_stats": normalization_stats,
        "stats_source": portable_source_name(stats_path),
        "data_source": portable_source_name(data_path),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def restore_rng_state(checkpoint):
    rng_state = checkpoint.get("rng_state")
    if not rng_state:
        return
    if rng_state.get("torch") is not None:
        torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def train_flow_egnn(
    data_path,
    output_dir="models",
    epochs=100,
    save_checkpoints=25,
    batch_size=64,
    k_knn=16,
    n_layers=3,
    hidden_dim=256,
    learning_rate=1e-3,
    model_load_path=None,
    device=None,
    max_nodes_per_batch=None,
    gradient_accumulation_steps=1,
    amp=False,
    num_workers=0,
    pin_memory=None,
    stats_path=None,
    seed=None,
):
    seed_everything(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    resume_checkpoint = None
    legacy_resume = False
    start_epoch = 1
    if model_load_path is not None:
        model_state_dict, resume_checkpoint = load_model_checkpoint(model_load_path)
        legacy_resume = resume_checkpoint is None
        if legacy_resume:
            resume_checkpoint = {"model_state_dict": model_state_dict}
        else:
            saved_model_config = resume_checkpoint.get("model_config", {})
            hidden_dim = saved_model_config.get("hidden_dim", hidden_dim)
            n_layers = saved_model_config.get("n_layers", n_layers)
            k_knn = saved_model_config.get("k_knn", k_knn)
            start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
            saved_training_config = resume_checkpoint.get("training_config", {})
            seed = saved_training_config.get("seed", seed)
            if saved_training_config.get("amp", False):
                amp = True

    model = FlowEGNN(hidden_dim=hidden_dim, n_layers=n_layers)
    if resume_checkpoint is not None:
        model.load_state_dict(model_state_dict)
    model.to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    if resume_checkpoint is not None and not legacy_resume:
        optimiser.load_state_dict(resume_checkpoint["optimizer_state_dict"])

    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if max_nodes_per_batch is not None and max_nodes_per_batch <= 0:
        raise ValueError("max_nodes_per_batch must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    device_type = torch.device(device).type
    amp_enabled = amp and device_type == "cuda"
    if amp and not amp_enabled:
        print("AMP requested but only enabled for CUDA; using float32")
    amp_dtype = torch.float16 if amp_enabled else None
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
    if (
        scaler is not None
        and resume_checkpoint is not None
        and resume_checkpoint.get("scaler_state_dict") is not None
    ):
        scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])

    embedded_stats = (
        resume_checkpoint.get("normalization_stats")
        if resume_checkpoint is not None
        else None
    )
    if stats_path is None and embedded_stats is not None:
        normalization_stats = embedded_stats
        resolved_stats_path = resume_checkpoint.get(
            "stats_source", "embedded-in-checkpoint"
        )
    else:
        normalization_stats, resolved_stats_path = load_normalization_stats(
            data_path=data_path,
            stats_path=stats_path,
        )

    model_config = {
        "edge_feat_dim": 15,
        "edge_out_dim": 11,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "k_knn": k_knn,
    }
    training_config = {
        "batch_size": batch_size,
        "max_nodes_per_batch": max_nodes_per_batch,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": optimiser.param_groups[0]["lr"],
        "amp": amp_enabled,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "seed": seed,
    }
    restore_rng_state(resume_checkpoint or {})

    dataset = load_graph_dataset(data_path)
    if len(dataset) == 0:
        raise ValueError(f"Training dataset is empty: {data_path}")

    if pin_memory is None:
        pin_memory = device_type == "cuda"
    training_config["pin_memory"] = pin_memory
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if max_nodes_per_batch is None:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            **loader_kwargs,
        )
    else:
        sampler = DynamicBatchSampler(
            dataset,
            max_num=max_nodes_per_batch,
            mode="node",
            shuffle=True,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            **loader_kwargs,
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if start_epoch > epochs:
        raise ValueError(
            f"Checkpoint already completed epoch {start_epoch - 1}, "
            f"but requested total epochs is {epochs}"
        )

    for epoch in range(start_epoch, epochs + 1):
        total_loss = 0.0
        loss_x = 0.0
        loss_ani = 0.0
        loss_w = 0.0

        num_batches = 0
        optimiser.zero_grad(set_to_none=True)
        for batch in loader:
            num_batches += 1
            step_optimizer = (
                num_batches % gradient_accumulation_steps == 0
            )

            metrics = training_step(
                model=model,
                batch=batch,
                optimiser=optimiser,
                device=device,
                K_knn=k_knn,
                loss_divisor=gradient_accumulation_steps,
                scaler=scaler,
                amp_dtype=amp_dtype,
                step_optimizer=step_optimizer,
                zero_grad=False,
                non_blocking=pin_memory,
            )

            total_loss += metrics["loss"].item()
            loss_x += metrics["loss_x"].item()
            loss_ani += metrics["loss_ani"].item()
            loss_w += metrics["loss_w"].item()

        if num_batches == 0:
            raise RuntimeError("The data loader produced no batches")

        remainder = num_batches % gradient_accumulation_steps
        if remainder:
            finish_optimizer_step(
                model,
                optimiser,
                scaler=scaler,
                gradient_multiplier=gradient_accumulation_steps / remainder,
            )

        avg_loss = total_loss / num_batches
        loss_x = loss_x / num_batches
        loss_ani = loss_ani / num_batches
        loss_w = loss_w / num_batches

        print(f"Epoch {epoch:4d} | Loss {avg_loss:.6f} | Loss_x {loss_x:.6f} | Loss_ani {loss_ani:.6f} | Loss_w {loss_w:.6f} ")
        if save_checkpoints > 0 and epoch % save_checkpoints == 0:
            checkpoint = output_dir / (
                f"model_{hidden_dim}hd_{n_layers}ly_{k_knn}knn_{epoch}.pt"
            )
            torch.save(
                make_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimiser=optimiser,
                    scaler=scaler,
                    model_config=model_config,
                    training_config=training_config,
                    normalization_stats=normalization_stats,
                    stats_path=resolved_stats_path,
                    data_path=data_path,
                ),
                checkpoint,
            )

    final_path = output_dir / (
        f"model_{hidden_dim}hd_{n_layers}ly_{k_knn}knn_{epochs}_final.pt"
    )
    torch.save(
        make_checkpoint(
            epoch=epochs,
            model=model,
            optimiser=optimiser,
            scaler=scaler,
            model_config=model_config,
            training_config=training_config,
            normalization_stats=normalization_stats,
            stats_path=resolved_stats_path,
            data_path=data_path,
        ),
        final_path,
    )
    if hasattr(dataset, "close"):
        dataset.close()
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the APD flow-matching model")

    parser.add_argument(
        "--data",
        required=True,
        help="Normalized .pt file or on-disk dataset directory",
    )
    parser.add_argument(
        "--stats",
        default=None,
        help="Normalization stats; inferred from generated dataset names",
    )
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_checkpoints", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--k-knn", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume a full checkpoint; legacy weights-only files also load",
    )
    parser.add_argument("--device", default=None, help="For example: cpu or cuda")
    parser.add_argument(
        "--max-nodes-per-batch",
        type=int,
        default=None,
        help="Use node-budget batching instead of a fixed graph count",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Use CUDA float16 AMP")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed Python, NumPy, and PyTorch random number generators",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pin loader memory; defaults to enabled on CUDA",
    )

    args = parser.parse_args()

    train_flow_egnn(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        save_checkpoints=args.save_checkpoints,
        batch_size=args.batch_size,
        k_knn=args.k_knn,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        model_load_path=args.resume,
        device=args.device,
        max_nodes_per_batch=args.max_nodes_per_batch,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        amp=args.amp,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        stats_path=args.stats,
        seed=args.seed,
    )
