"""Scene selection for segmentation + depth fusion — reuses mvs_scenes and
keeps only scenes where BOTH depth-map kinds (.geometric.bin and
.photometric.bin) exist. Scenes with only .geometric.bin are skipped.
"""
import mvs_scenes

FusionScene = mvs_scenes.MvsScene   # same shape; workspace/image_dir/n_frames all apply


def has_both_bin_kinds(workspace) -> bool:
    d = workspace / "dense" / "stereo" / "depth_maps"
    if not d.exists():
        return False
    return any(d.glob("*.geometric.bin")) and any(d.glob("*.photometric.bin"))


def load_scenes(scenes_filter: list[str] | None = None) -> tuple[list, list[str]]:
    """Returns (eligible_scenes, skipped_scene_names). eligible sorted by
    frame count descending, same convention as the rest of the pipeline."""
    all_scenes = mvs_scenes.load_scenes(
        scenes_filter=scenes_filter, status_filter=["EXCELLENT", "GOOD"])
    eligible, skipped = [], []
    for s in all_scenes:
        if has_both_bin_kinds(s.workspace):
            eligible.append(s)
        else:
            skipped.append(s.name)
    eligible.sort(key=lambda s: -s.n_frames)
    return eligible, skipped


def array_split(scenes: list, task_id: int, n_tasks: int) -> list:
    return [s for i, s in enumerate(scenes) if i % n_tasks == task_id]
