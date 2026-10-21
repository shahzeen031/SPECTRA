"""Scene selection for the MVS stage — driven by the SfM master CSV."""
import csv
from dataclasses import dataclass
from pathlib import Path

import mvs_config as cfg


@dataclass
class MvsScene:
    name: str
    sfm_status: str
    n_frames: int
    image_dir: Path         # original rgb dir (undistort input)
    sfm_dir: Path           # sfm/output/{scene}

    @property
    def workspace(self) -> Path:
        return cfg.OUTPUT_ROOT / self.name


def load_scenes(scenes_filter: list[str] | None = None,
                status_filter: list[str] | None = None) -> list[MvsScene]:
    """All MVS-eligible scenes, sorted by frame count DESC so monsters start
    first and don't tail-end the batch."""
    inventory = {}
    inv_csv = cfg.SFM_OUTPUT_ROOT / "scene_inventory.csv"
    with open(inv_csv) as f:
        for row in csv.DictReader(f):
            inventory[row["scene"]] = row["image_dir"]

    out = []
    with open(cfg.SFM_MASTER_CSV) as f:
        for row in csv.DictReader(f):
            status = row["final_status"]
            if status not in cfg.ALLOWED_STATUS:
                continue  # skips SFM_FAIL and TOO_SMALL
            if status_filter and status not in status_filter:
                continue
            if scenes_filter and row["scene"] not in scenes_filter:
                continue
            out.append(MvsScene(
                name=row["scene"],
                sfm_status=status,
                n_frames=int(row["n_frames"]),
                image_dir=Path(inventory[row["scene"]]),
                sfm_dir=cfg.SFM_OUTPUT_ROOT / row["scene"],
            ))
    out.sort(key=lambda s: -s.n_frames)
    return out


def array_split(scenes: list[MvsScene], task_id: int,
                n_tasks: int) -> list[MvsScene]:
    """Round-robin over the size-DESC list -> balanced heavy/light mix."""
    return [s for i, s in enumerate(scenes) if i % n_tasks == task_id]
