"""
Local collision MVP training — runs on CPU/MPS without PyTorch Lightning.
Logs every 100 steps, saves checkpoints every 1000 steps.

Usage:
    python scripts/train_collision_local.py --epochs 1
    python scripts/train_collision_local.py --epochs 1 --max_steps 500  # quick test
"""
import os, sys, time, json, argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
from navsim.agents.goalflow.goalflow_agent_collision import GoalFlowCollisionAgent
from navsim.agents.goalflow.goalflow_loss import goalflow_loss
from navsim.planning.training.dataset import CacheOnlyDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=0, help="0 = full epoch")
    parser.add_argument("--cache_path", type=str, default=str(ROOT / "exp/feature_cache_test"))
    parser.add_argument("--save_dir", type=str, default=str(ROOT / "exp/collision_mvp_local"))
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Model
    config = GoalFlowConfig(
        training=True, has_navi=True, start=True,
        freeze_perception=True, tf_d_model=1024,
        voc_path=str(ROOT / "data/cluster_points_8192_.npy"),
        trajectory_weight=50.0, agent_class_weight=0.2,
        agent_box_weight=0.05, bev_semantic_weight=0.2,
        agent_loss=True, adv_mode=True,
    )
    agent = GoalFlowCollisionAgent(config=config, lr=args.lr,
        checkpoint_path=str(ROOT / "data/goalflow_traj_epoch_54-step_18260.ckpt"))
    sd = torch.load(str(ROOT / "data/goalflow_traj_epoch_54-step_18260.ckpt"), map_location="cpu")["state_dict"]
    agent.load_state_dict({k.replace("agent.", ""): v for k, v in sd.items()}, strict=False)
    agent.train()

    # Dataset — num_workers=0 for macOS compatibility
    dataset = CacheOnlyDataset(
        cache_path=args.cache_path,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, agent._goalflow_model.parameters()), lr=args.lr
    )

    steps_per_epoch = len(dataset) // args.batch_size
    max_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch

    print(f"{'='*60}")
    print(f"Collision MVP Training")
    print(f"  Dataset:    {len(dataset)} scenes")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Steps/epoch:{steps_per_epoch}")
    print(f"  Max steps:  {max_steps}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Save dir:   {save_dir}")
    print(f"{'='*60}")
    print(flush=True)

    loss_log = []
    global_step = 0

    for epoch in range(args.epochs):
        t_epoch = time.time()
        for i, (features, targets) in enumerate(loader):
            if i >= max_steps:
                break

            # Fix token format
            if isinstance(features.get("token"), (list, tuple)):
                pass
            else:
                features["token"] = [features["token"]]

            t0 = time.time()
            predictions = agent.forward(features, targets)
            loss_dict = goalflow_loss(targets, predictions, config)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dt = time.time() - t0

            traj_loss = loss_dict["trajectory_loss"].item()
            loss_log.append({"epoch": epoch, "step": global_step, "loss": loss.item(), "traj_loss": traj_loss, "time": dt})
            global_step += 1

            if i % 100 == 0:
                elapsed = time.time() - t_epoch
                eta = (elapsed / (i + 1)) * (max_steps - i) / 60
                print(f"  E{epoch} [{i:5d}/{max_steps}] loss={loss.item():.4f} traj={traj_loss:.4f} "
                      f"| {dt:.2f}s/step | {elapsed/60:.1f}min, ~{eta:.0f}min left", flush=True)

            if (global_step) % 1000 == 0:
                ckpt = save_dir / f"collision_step_{global_step}.pt"
                torch.save({"step": global_step, "state_dict": {f"agent.{k}": v for k, v in agent.state_dict().items()}}, ckpt)
                print(f"  → Checkpoint: {ckpt}", flush=True)

        # End of epoch
        elapsed = time.time() - t_epoch
        print(f"\n  Epoch {epoch} done: {min(i+1, max_steps)} steps in {elapsed/60:.1f} min\n", flush=True)

    # Final save
    ckpt = save_dir / f"collision_final.pt"
    torch.save({"step": global_step, "state_dict": {f"agent.{k}": v for k, v in agent.state_dict().items()}}, ckpt)

    with open(save_dir / "loss_log.json", "w") as f:
        json.dump(loss_log, f)

    # Summary
    import numpy as np
    traj_losses = [x["traj_loss"] for x in loss_log]
    n = min(100, len(traj_losses) // 4)
    if n > 0:
        start_avg = np.mean(traj_losses[:n])
        end_avg = np.mean(traj_losses[-n:])
        print(f"\n{'='*60}")
        print(f"DONE — {global_step} steps")
        print(f"Trajectory loss: {start_avg:.4f} (start) → {end_avg:.4f} (end)")
        print(f"Change: {(end_avg - start_avg) / start_avg * 100:+.1f}%")
        print(f"Checkpoint: {ckpt}")
        print(f"Loss log:   {save_dir / 'loss_log.json'}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
