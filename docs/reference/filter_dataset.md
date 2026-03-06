# filter_dataset.py

Filter a LeRobot dataset by keeping or excluding specific episodes. lerobot-env venv.

Creates a new dataset with renumbered episodes. **Does not modify the original** (unless `--delete-original` is passed).

---

## CLI

```bash
# Exclude specific episodes
python scripts/tools/filter_dataset.py \
    --input  data/recordings/v1 \
    --output data/recordings/v1_filtered \
    --exclude-episodes 3 7 12

# Keep only specific episodes
python scripts/tools/filter_dataset.py \
    --input  data/recordings/v1 \
    --output data/recordings/v1_filtered \
    --keep-episodes 0 1 5

# Delete original after successful filtering (prompts for confirmation)
python scripts/tools/filter_dataset.py \
    --input  data/recordings/v1 \
    --output data/recordings/v1_filtered \
    --exclude-episodes 3 7 --delete-original
```

| Flag | Description |
|------|-------------|
| `--input` / `-i` | Input dataset directory (required) |
| `--output` / `-o` | Output dataset directory (required, must not exist) |
| `--keep-episodes` / `-k` | Episode indices to keep (all others removed) |
| `--exclude-episodes` / `-e` | Episode indices to remove (all others kept) |
| `--delete-original` / `-d` | Delete input dataset after filtering (requires typing `DELETE`) |

`--keep-episodes` and `--exclude-episodes` are mutually exclusive; exactly one is required.

---

## What Gets Created

Output dataset is a complete LeRobot v3.0 dataset:

```
output/
├── data/chunk-000/file-000.parquet     # filtered + renumbered frames
├── meta/
│   ├── info.json                       # updated total_episodes, total_frames
│   ├── stats.json                      # recomputed for observation.state, action
│   ├── tasks.parquet                   # copied from input
│   ├── episode_metadata.json           # remapped episode indices (if present)
│   └── episodes/chunk-000/file-000.parquet  # updated lengths, timestamps, frame offsets
└── videos/
    └── observation.images.{top,wrist}/chunk-000/file-000.mp4  # re-extracted + concatenated
```

Episode indices are remapped sequentially: e.g. keeping episodes [0, 3, 7] → new indices [0, 1, 2].

---

## Notes

- **Videos** are always re-processed: each episode's segment is extracted via `ffmpeg -c copy` (lossless), then all segments are concatenated into a single `file-000.mp4`.
- **stats.json** recomputes only `observation.state` and `action` — image stats are not recomputed.
- Output directory must not exist; the script errors if it does.
- `--delete-original` prompts for manual confirmation (`DELETE`) before deleting.
