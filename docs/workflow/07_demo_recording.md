# Recording Demo Media

How to record eval rollouts and teleoperation for README / documentation.

---

## Setup

- **Screen recorder:** OBS Studio (`sudo apt install obs-studio`)
- **Conversion:** ffmpeg

---

## Eval rollout

```bash
eval "$(./activate_isaaclab.sh)"
python scripts/eval/eval_vla_policy.py \
    --checkpoint <path_to_pretrained_model> \
    --env figure_shape_placement \
    --episode-seed <seed> \
    --preview --start-paused \
    --auto-server \
    --no-domain-rand \
    --max-steps 600
```

Key flags:
- `--episode-seed` — reproduce a specific scene (find seeds in `outputs/eval_sweeps/*/summary.json`)
- `--start-paused` — waits for Space press before starting (time to set up OBS)
- `--preview` — opens camera preview window (top + wrist side-by-side)
- `--no-domain-rand` — clean scene, no visual noise
- `--auto-server` — auto-starts gRPC policy server

Record the preview window in OBS, press Space in viewer to start the episode.

## Teleoperation

```bash
eval "$(./activate_isaaclab.sh)"
python scripts/teleop/record_episodes.py \
    --output tmp/recordings_tmp \
    --env figure_shape_placement \
    --teleop-device=so101leader \
    --calibration-file=leader_1.json \
    --port=/dev/ttyACM0 \
    --reward-mode success
```

Without `--headless` the Isaac Sim GUI viewport opens — record that window in OBS.
Add `--preview` to also show the camera preview window.

### Full game demo scene

For visually rich teleop demos with all 9 shape sorting figures on the platform:

```bash
eval "$(./activate_isaaclab.sh)"
python scripts/teleop/record_episodes.py \
    --output tmp/recordings_demo \
    --env full_game_demo \
    --teleop-device=so101leader \
    --calibration-file=leader_1.json \
    --port=/dev/ttyACM0 \
    --reward-mode success
```

The `full_game_demo` env spawns all figures (cube, star, hexagon, cylinder, diamond, clover, X, parallelepiped, triangle prism) at their slots on the platform. Physics runs at 120 Hz for stability with many objects.

---

## Post-processing

### Trim

```bash
# Trim by time (seconds) — re-encodes for frame-accurate cuts
ffmpeg -ss <start> -i input.mp4 -t <duration> output_trimmed.mp4

# Note: adding -c copy is faster but cuts only at keyframes (imprecise)
```

### Convert to GIF

```bash
# Single video — crop black borders, scale, optimize palette
ffmpeg -i input_trimmed.mp4 \
    -vf "crop=W:H:X:Y,fps=10,scale=540:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" \
    output.gif

# Detect black borders automatically
ffmpeg -i input.mp4 -vf "cropdetect=24:16:0" -frames:v 50 -f null - 2>&1 | grep cropdetect
```

### Side-by-side comparison GIF

```bash
# 3 videos → one GIF, synced playback, with text labels
# tpad pads shorter videos with last frame to match longest
ffmpeg -y \
    -i video1.mp4 -i video2.mp4 -i video3.mp4 \
    -filter_complex "
      [0:v]crop=W:H:X:Y,fps=10,scale=400:-1:flags=lanczos,
        drawtext=text='Label 1':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=16:fontcolor=white:borderw=2:bordercolor=black:x=(w-text_w)/2:y=8[v0];
      [1:v]crop=W:H:X:Y,fps=10,scale=400:-1:flags=lanczos,tpad=stop_mode=clone:stop_duration=<pad_seconds>,
        drawtext=text='Label 2':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=16:fontcolor=white:borderw=2:bordercolor=black:x=(w-text_w)/2:y=8[v1];
      [2:v]crop=W:H:X:Y,fps=10,scale=400:-1:flags=lanczos,tpad=stop_mode=clone:stop_duration=<pad_seconds>,
        drawtext=text='Label 3':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=16:fontcolor=white:borderw=2:bordercolor=black:x=(w-text_w)/2:y=8[v2];
      [v0][v1][v2]hstack=inputs=3[out];
      [out]split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3
    " output_comparison.gif
```

### Size guidelines

- GitHub README: keep GIFs under 10 MB
- `scale=540:-1` + `fps=10` + `max_colors=128` is a good balance (~4-7 MB for 15-30s)
- Reduce `scale` or `fps` if still too large
