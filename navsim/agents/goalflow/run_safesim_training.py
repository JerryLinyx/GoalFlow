"""
Flexible SafeSim training entrypoint supporting:
  - from-scratch baseline
  - GoalFlow FM-head transfer sanity runs
  - Stage 1 original-data prior alignment
  - Stage 2 filtered-data fine-tune with mixed replay
"""

import argparse
import json
import os
import pathlib
import random
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navsim.agents.goalflow.safesim_agent import SafeSimAgent
from navsim.agents.goalflow.safesim_config import SafeSimConfig
from navsim.agents.goalflow.safesim_dataset import SafeSimTemporalDataset, safesim_collate_fn


def register_checkpoint_safe_globals():
    """
    PyTorch 2.6 tightened torch.load defaults through weights_only=True in some
    downstream callers. Lightning resume checkpoints may still contain trusted
    config/path objects, so we allowlist the ones produced by this project.
    """
    if not hasattr(torch.serialization, "add_safe_globals"):
        return
    torch.serialization.add_safe_globals([SafeSimConfig, pathlib.PosixPath])


def split_dataset_random(dataset: SafeSimTemporalDataset, val_split: float, seed: int = 0):
    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError(f"Dataset has {dataset_size} samples; need >= 2 for train/val split.")
    val_size = max(1, int(round(dataset_size * val_split)))
    val_size = min(val_size, dataset_size - 1)
    train_size = dataset_size - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def resolve_runtime(requested_gpus: int, requested_precision: str):
    if requested_gpus > 0 and torch.cuda.is_available():
        return "gpu", requested_gpus, requested_precision

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        precision = requested_precision if requested_precision not in {"16-mixed", "bf16-mixed"} else "32"
        return "mps", 1, precision

    precision = requested_precision if requested_precision not in {"16-mixed", "bf16-mixed"} else "32"
    return "cpu", 1, precision


def build_dataset(config: SafeSimConfig, hdf5_paths, split: str, target_policy: str) -> SafeSimTemporalDataset:
    return SafeSimTemporalDataset(
        config,
        hdf5_paths=hdf5_paths,
        split=split,
        target_policy=target_policy,
    )


def build_mixed_sampler(primary_len: int, replay_len: int, replay_ratio: float) -> WeightedRandomSampler:
    primary_ratio = 1.0 - replay_ratio
    weights = torch.cat([
        torch.full((primary_len,), primary_ratio / max(primary_len, 1), dtype=torch.double),
        torch.full((replay_len,), replay_ratio / max(replay_len, 1), dtype=torch.double),
    ])
    return WeightedRandomSampler(weights=weights, num_samples=primary_len + replay_len, replacement=True)


def load_safe_checkpoint(agent: SafeSimAgent, checkpoint_path: str, strict: bool = True):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    if strict:
        agent.load_state_dict(state_dict, strict=True)
        return

    model_state = agent.state_dict()
    compatible_state = {}
    skipped_keys = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped_keys.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_keys.append(key)
            continue
        compatible_state[key] = value

    model_state.update(compatible_state)
    agent.load_state_dict(model_state, strict=True)
    print(
        f"[SafeSim] Loaded checkpoint partially from {checkpoint_path}: "
        f"{len(compatible_state)} compatible keys, {len(skipped_keys)} skipped."
    )


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description='SafeSim Dangerous Trajectory Training')

    parser.add_argument('--hdf5_paths', nargs='+', required=True, help='Primary HDF5 data files')
    parser.add_argument('--replay_hdf5_paths', nargs='*', default=[], help='Replay HDF5 files for mixed Stage-2 training')
    parser.add_argument('--val_split', type=float, default=0.1)

    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--log_dir', type=str, default='./safesim_logs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_every_n_epochs', type=int, default=5)
    parser.add_argument('--limit_train_batches', type=float, default=1.0)
    parser.add_argument('--limit_val_batches', type=float, default=1.0)

    parser.add_argument('--tf_d_model', type=int, default=256)
    parser.add_argument('--tf_num_layers', type=int, default=3)
    parser.add_argument('--max_other_agents', type=int, default=10)
    parser.add_argument('--anchor_size', type=int, default=10)
    parser.add_argument('--infer_steps', type=int, default=100)
    parser.add_argument('--temporal_stride', type=int, default=5)
    parser.add_argument('--history_len', type=int, default=4)
    parser.add_argument('--history_stride', type=int, default=5)
    parser.add_argument('--cfg_scale', type=float, default=1.0)
    parser.add_argument('--condition_dropout_prob', type=float, default=0.15)
    parser.add_argument('--use_goal_condition', type=int, default=0)
    parser.add_argument('--target_policy', type=str, default='raw_gt', choices=['raw_gt', 'action', 'nearest_action_sample'])
    parser.add_argument('--replay_ratio', type=float, default=0.3)
    parser.add_argument('--trajectory_weight', type=float, default=1.0)
    parser.add_argument('--terminal_xy_weight', type=float, default=0.0)
    parser.add_argument('--terminal_heading_weight', type=float, default=0.0)
    parser.add_argument('--ctrl_softmin_weight', type=float, default=0.0)
    parser.add_argument('--ctrl_softmin_beta', type=float, default=4.0)

    parser.add_argument('--init_checkpoint', type=str, default=None, help='Original GoalFlow checkpoint for FM transfer')
    parser.add_argument('--init_mode', type=str, default='none', choices=['none', 'fm_head_conservative', 'fm_head_extended'])
    parser.add_argument('--freeze_loaded_fm_epochs', type=int, default=0)
    parser.add_argument('--model_checkpoint', type=str, default=None, help='SafeSim checkpoint used as model initialization')
    parser.add_argument('--checkpoint', type=str, default=None, help='Trainer resume checkpoint')
    parser.add_argument('--stage_name', type=str, default='baseline')
    args = parser.parse_args()

    if args.model_checkpoint and args.init_mode != 'none':
        raise ValueError("Use either --model_checkpoint or --init_mode/--init_checkpoint, not both.")

    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    register_checkpoint_safe_globals()

    config = SafeSimConfig(
        hdf5_paths=args.hdf5_paths,
        lr=args.lr,
        tf_d_model=args.tf_d_model,
        tf_num_layers=args.tf_num_layers,
        max_other_agents=args.max_other_agents,
        anchor_size=args.anchor_size,
        infer_steps=args.infer_steps,
        temporal_stride=args.temporal_stride,
        history_len=args.history_len,
        history_stride=args.history_stride,
        cfg_scale=args.cfg_scale,
        condition_dropout_prob=args.condition_dropout_prob,
        use_goal_condition=bool(args.use_goal_condition),
        target_policy=args.target_policy,
        trajectory_weight=args.trajectory_weight,
        terminal_xy_weight=args.terminal_xy_weight,
        terminal_heading_weight=args.terminal_heading_weight,
        ctrl_softmin_weight=args.ctrl_softmin_weight,
        ctrl_softmin_beta=args.ctrl_softmin_beta,
        freeze_loaded_fm_epochs=args.freeze_loaded_fm_epochs,
        init_mode=args.init_mode,
        init_checkpoint=args.init_checkpoint or "",
        model_checkpoint=args.model_checkpoint or "",
        stage_name=args.stage_name,
        training=True,
    )

    accelerator, devices, precision = resolve_runtime(args.gpus, args.precision)
    effective_num_workers = args.num_workers if accelerator != "cpu" else 0
    pin_memory = accelerator != "cpu"

    primary_dataset = build_dataset(config, args.hdf5_paths, split='train', target_policy=args.target_policy)
    primary_train, primary_val = split_dataset_random(primary_dataset, args.val_split, seed=args.seed)

    train_dataset = primary_train
    train_sampler = None
    train_shuffle = True

    if args.replay_hdf5_paths:
        replay_dataset = build_dataset(config, args.replay_hdf5_paths, split='train', target_policy='raw_gt')
        replay_train, replay_val = split_dataset_random(replay_dataset, args.val_split, seed=args.seed)
        train_dataset = ConcatDataset([primary_train, replay_train])
        train_sampler = build_mixed_sampler(len(primary_train), len(replay_train), args.replay_ratio)
        train_shuffle = False
        replay_summary = {
            "replay_train_samples": len(replay_train),
            "replay_val_samples": len(replay_val),
            "replay_ratio": args.replay_ratio,
            "replay_paths": args.replay_hdf5_paths,
        }
    else:
        replay_summary = {}

    print(f"[SafeSim] Primary train: {len(primary_train)}, val: {len(primary_val)}")
    if args.replay_hdf5_paths:
        print(f"[SafeSim] Replay train mixed in: {replay_summary['replay_train_samples']} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=effective_num_workers,
        collate_fn=safesim_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        primary_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        collate_fn=safesim_collate_fn,
        pin_memory=pin_memory,
    )

    model = SafeSimAgent(config)
    transfer_report = {}

    if args.model_checkpoint:
        load_safe_checkpoint(model, args.model_checkpoint, strict=not config.use_goal_condition)

    if args.init_mode != 'none':
        if not args.init_checkpoint:
            raise ValueError("--init_checkpoint is required when --init_mode is enabled.")
        transfer_report = model.model.load_goalflow_fm_head(args.init_checkpoint, args.init_mode)
        if not transfer_report["loaded_modules"]:
            source_hint = transfer_report.get("source_tf_d_model_hint")
            raise RuntimeError(
                "No GoalFlow FM-head weights were loaded. "
                f"Requested mode={args.init_mode}, current tf_d_model={args.tf_d_model}, "
                f"checkpoint hidden dim hint={source_hint}. "
                "If you intend to transfer from the released GoalFlow checkpoint, run with --tf_d_model 1024."
            )
        if args.freeze_loaded_fm_epochs > 0:
            model.model.freeze_loaded_transfer_modules()

    log_dir = Path(args.log_dir)
    split_summary = {
        "seed": args.seed,
        "val_split": args.val_split,
        "stage_name": args.stage_name,
        "target_policy": args.target_policy,
        "primary_train_samples": len(primary_train),
        "primary_val_samples": len(primary_val),
        "primary_paths": args.hdf5_paths,
        "tf_d_model": args.tf_d_model,
        "temporal_stride": args.temporal_stride,
        "cfg_scale": args.cfg_scale,
        "condition_dropout_prob": args.condition_dropout_prob,
        "use_goal_condition": bool(args.use_goal_condition),
        "trajectory_weight": args.trajectory_weight,
        "terminal_xy_weight": args.terminal_xy_weight,
        "terminal_heading_weight": args.terminal_heading_weight,
        "ctrl_softmin_weight": args.ctrl_softmin_weight,
        "ctrl_softmin_beta": args.ctrl_softmin_beta,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "freeze_loaded_fm_epochs": args.freeze_loaded_fm_epochs,
        "init_mode": args.init_mode,
        "init_checkpoint": args.init_checkpoint,
        "model_checkpoint": args.model_checkpoint,
    }
    split_summary.update(replay_summary)
    write_json(log_dir / "split_summary.json", split_summary)
    if transfer_report:
        write_json(log_dir / "transfer_report.json", transfer_report)

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.log_dir, 'checkpoints'),
            filename='safesim-{epoch:02d}-{val_loss:.4f}',
            every_n_epochs=args.save_every_n_epochs,
            save_last=True,
            save_top_k=-1,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=os.path.join(args.log_dir, 'checkpoints'),
            filename='best-val-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    logger = CSVLogger(save_dir=args.log_dir, name='csv_logs')

    print(
        f"[SafeSim] Runtime: accelerator={accelerator}, devices={devices}, precision={precision}, "
        f"num_workers={effective_num_workers}, pin_memory={pin_memory}"
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        default_root_dir=args.log_dir,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=1.0,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.checkpoint,
    )


if __name__ == '__main__':
    main()
