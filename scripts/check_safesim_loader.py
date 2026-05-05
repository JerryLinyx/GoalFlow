#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navsim.safesim import SafeSimDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the Safe-Sim HDF5 loader.")
    parser.add_argument(
        "--hdf5",
        type=str,
        default="safesim/case1_best/data.hdf5",
        help="Path to a Safe-Sim HDF5 file.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index to inspect.",
    )
    args = parser.parse_args()

    dataset = SafeSimDataset(args.hdf5)
    print(f"dataset_size={len(dataset)}")

    sample = dataset[args.index]
    print(f"key={sample['key']}")
    print(f"scene_name={sample['scene_name']}")
    print(f"case_id={int(sample['case_id'])}")
    print(f"ego_idx={int(sample['ego_idx'])} ctrl_idx={int(sample['ctrl_idx'])}")
    print(f"num_agents={int(sample['num_agents'])} num_steps={int(sample['num_steps'])}")
    print(f"action_horizon={int(sample['action_horizon'])}")
    print(f"num_action_samples={int(sample['num_action_samples'])}")
    print(f"centroid_shape={tuple(sample['centroid'].shape)}")
    print(f"drivable_map_shape={tuple(sample['drivable_map'].shape)}")
    print(f"action_valid_steps={sample['replan_mask'].nonzero().flatten().tolist()}")
    print(f"track_ids={sample['track_id'].tolist()}")

    dataset.close()


if __name__ == "__main__":
    main()
