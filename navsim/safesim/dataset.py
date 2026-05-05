from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import h5py
import numpy as np
import torch


SCENE_KEY_PATTERN = re.compile(
    r"^(?P<scene_name>scene-\d+)_ego_(?P<ego_idx>\d+)_ctrl_\[(?P<ctrl_idx>\d+)\]_(?P<variant_idx>\d+)$"
)
CASE_ID_PATTERN = re.compile(r"case(?P<case_id>\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class SafeSimSceneKey:
    raw_key: str
    scene_name: str
    ego_idx: int
    ctrl_idx: int
    variant_idx: int
    case_id: Optional[int] = None

    @classmethod
    def parse(cls, raw_key: str, case_id: Optional[int] = None) -> "SafeSimSceneKey":
        match = SCENE_KEY_PATTERN.match(raw_key)
        if match is None:
            raise ValueError(f"Invalid Safe-Sim key: {raw_key}")

        return cls(
            raw_key=raw_key,
            scene_name=match.group("scene_name"),
            ego_idx=int(match.group("ego_idx")),
            ctrl_idx=int(match.group("ctrl_idx")),
            variant_idx=int(match.group("variant_idx")),
            case_id=case_id,
        )


@dataclass(frozen=True)
class SafeSimSceneRecord:
    hdf5_path: Path
    scene_key: SafeSimSceneKey


class SafeSimDataset(torch.utils.data.Dataset):
    """
    Lazy HDF5 dataset for Safe-Sim rollouts.

    Each item is returned as a flat dictionary containing:
      - raw scene tensors converted to torch tensors
      - parsed metadata from the HDF5 key
      - validity masks for sparse action fields

    Notes about the observed Safe-Sim export format:
      - `action_positions` / `action_yaws` only contain finite values on replanning
        steps (for example every 5 frames). We sanitize invalid values to zero and
        expose `action_valid_mask` so downstream code can ignore padded plans.
      - `scene_index` / `track_id` may be stored as [N, T] even though they are
        time-invariant. We collapse them to [N] after checking they are constant.
    """

    def __init__(
        self,
        hdf5_paths: Union[str, Path, Sequence[Union[str, Path]]],
        scene_names: Optional[Iterable[str]] = None,
        sanitize_actions: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(hdf5_paths, (str, Path)):
            paths = [Path(hdf5_paths)]
        else:
            paths = [Path(path) for path in hdf5_paths]

        if not paths:
            raise ValueError("SafeSimDataset requires at least one HDF5 path.")

        self._paths = [path.expanduser().resolve() for path in paths]
        self._sanitize_actions = sanitize_actions
        self._scene_names = set(scene_names) if scene_names is not None else None
        self._records = self._index_records(self._paths, self._scene_names)
        self._open_files: Dict[Path, h5py.File] = {}

    @staticmethod
    def _infer_case_id(path: Path) -> Optional[int]:
        for part in reversed(path.parts):
            match = CASE_ID_PATTERN.search(part)
            if match is not None:
                return int(match.group("case_id"))
        return None

    @classmethod
    def _index_records(
        cls,
        paths: Sequence[Path],
        scene_names: Optional[set[str]],
    ) -> List[SafeSimSceneRecord]:
        records: List[SafeSimSceneRecord] = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Safe-Sim HDF5 file not found: {path}")

            case_id = cls._infer_case_id(path)
            with h5py.File(path, "r") as h5_file:
                for raw_key in sorted(h5_file.keys()):
                    scene_key = SafeSimSceneKey.parse(raw_key, case_id=case_id)
                    if scene_names is not None and scene_key.scene_name not in scene_names:
                        continue
                    records.append(SafeSimSceneRecord(hdf5_path=path, scene_key=scene_key))

        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Union[str, torch.Tensor]]:
        record = self._records[index]
        h5_file = self._get_file(record.hdf5_path)
        group = h5_file[record.scene_key.raw_key]

        sample = self._load_group(group)
        sample.update(
            {
                "key": record.scene_key.raw_key,
                "scene_name": record.scene_key.scene_name,
                "hdf5_path": str(record.hdf5_path),
                "case_id": torch.tensor(
                    -1 if record.scene_key.case_id is None else record.scene_key.case_id,
                    dtype=torch.long,
                ),
                "ego_idx": torch.tensor(record.scene_key.ego_idx, dtype=torch.long),
                "ctrl_idx": torch.tensor(record.scene_key.ctrl_idx, dtype=torch.long),
                "variant_idx": torch.tensor(record.scene_key.variant_idx, dtype=torch.long),
            }
        )
        sample["num_agents"] = torch.tensor(sample["centroid"].shape[0], dtype=torch.long)
        sample["num_steps"] = torch.tensor(sample["centroid"].shape[1], dtype=torch.long)
        sample["action_horizon"] = torch.tensor(sample["action_positions"].shape[2], dtype=torch.long)
        sample["num_action_samples"] = torch.tensor(
            sample["action_sample_positions"].shape[2], dtype=torch.long
        )
        return sample

    @property
    def records(self) -> List[SafeSimSceneRecord]:
        return list(self._records)

    def get_record(self, index: int) -> SafeSimSceneRecord:
        return self._records[index]

    def close(self) -> None:
        for h5_file in self._open_files.values():
            h5_file.close()
        self._open_files.clear()

    def __del__(self) -> None:
        self.close()

    def _get_file(self, path: Path) -> h5py.File:
        h5_file = self._open_files.get(path)
        if h5_file is None:
            h5_file = h5py.File(path, "r")
            self._open_files[path] = h5_file
        return h5_file

    @staticmethod
    def _collapse_static_index(array: np.ndarray, name: str) -> np.ndarray:
        if array.ndim == 1:
            return array

        if array.ndim != 2:
            raise ValueError(f"{name} must be [N] or [N, T], got shape {array.shape}")

        if not np.all(array == array[:, :1]):
            raise ValueError(f"{name} is expected to be constant over time, got varying values")

        return array[:, 0]

    @staticmethod
    def _to_float_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(array, dtype=np.float32))

    @staticmethod
    def _to_bool_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(array, dtype=bool))

    def _sanitize_action_tensor(
        self,
        positions: np.ndarray,
        yaws: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reduce_axes = tuple(range(2, positions.ndim))
        valid_mask = np.isfinite(positions).all(axis=reduce_axes) & np.isfinite(yaws).all(
            axis=tuple(range(2, yaws.ndim))
        )

        if not self._sanitize_actions:
            return positions, yaws, valid_mask

        positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
        yaws = np.nan_to_num(yaws, nan=0.0, posinf=0.0, neginf=0.0)
        return positions, yaws, valid_mask

    @staticmethod
    def _extract_replan_mask(action_valid_mask: np.ndarray) -> np.ndarray:
        if action_valid_mask.ndim != 2:
            raise ValueError(
                f"action_valid_mask is expected to be [N, T], got shape {action_valid_mask.shape}"
            )
        return action_valid_mask.any(axis=0)

    def _load_group(self, group: h5py.Group) -> Dict[str, torch.Tensor]:
        centroid = self._to_float_tensor(group["centroid"][:])
        yaw = self._to_float_tensor(group["yaw"][:])
        curr_speed = self._to_float_tensor(group["curr_speed"][:])
        extent = self._to_float_tensor(group["extent"][:])

        action_positions_np = np.asarray(group["action_positions"][:], dtype=np.float32)
        action_yaws_np = np.asarray(group["action_yaws"][:], dtype=np.float32)
        action_positions_np, action_yaws_np, action_valid_mask_np = self._sanitize_action_tensor(
            action_positions_np,
            action_yaws_np,
        )

        action_sample_positions_np = np.asarray(group["action_sample_positions"][:], dtype=np.float32)
        action_sample_yaws_np = np.asarray(group["action_sample_yaws"][:], dtype=np.float32)
        (
            action_sample_positions_np,
            action_sample_yaws_np,
            action_sample_valid_mask_np,
        ) = self._sanitize_action_tensor(action_sample_positions_np, action_sample_yaws_np)

        drivable_map = self._to_bool_tensor(group["drivable_map"][:])
        exceed_lane = self._to_bool_tensor(group["exceed_lane"][:] > 0.5)
        raster_from_world = self._to_float_tensor(group["raster_from_world"][:])
        world_from_agent = self._to_float_tensor(group["world_from_agent"][:])

        scene_index = torch.from_numpy(
            self._collapse_static_index(np.asarray(group["scene_index"][:]), "scene_index").astype(np.int64)
        )
        track_id = torch.from_numpy(
            self._collapse_static_index(np.asarray(group["track_id"][:]), "track_id").astype(np.int64)
        )

        action_valid_mask = torch.from_numpy(action_valid_mask_np.astype(bool))
        action_sample_valid_mask = torch.from_numpy(action_sample_valid_mask_np.astype(bool))
        replan_mask = torch.from_numpy(self._extract_replan_mask(action_valid_mask_np).astype(bool))

        return {
            "centroid": centroid,
            "yaw": yaw,
            "curr_speed": curr_speed,
            "extent": extent,
            "action_positions": self._to_float_tensor(action_positions_np),
            "action_yaws": self._to_float_tensor(action_yaws_np),
            "action_sample_positions": self._to_float_tensor(action_sample_positions_np),
            "action_sample_yaws": self._to_float_tensor(action_sample_yaws_np),
            "action_valid_mask": action_valid_mask,
            "action_sample_valid_mask": action_sample_valid_mask,
            "replan_mask": replan_mask,
            "drivable_map": drivable_map,
            "exceed_lane": exceed_lane,
            "raster_from_world": raster_from_world,
            "world_from_agent": world_from_agent,
            "scene_index": scene_index,
            "track_id": track_id,
        }
