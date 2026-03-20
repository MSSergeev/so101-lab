# Tests

Tests are split into two directories based on dependencies:

## `tests/` — unit tests (no Isaac Sim required)

Run with:
```bash
pytest tests/
```

Cover devices, data recording, teleoperation logic. No simulator needed — fast, runnable in any environment.

## `tests/sim/` — functional tests (Isaac Sim required)

Run individually:
```bash
python tests/sim/test_template_env.py --gui
python tests/sim/test_rl_gym_wrapper.py
```

Require a running Isaac Sim environment (`eval "$(./activate_isaaclab.sh)"`). Test environment spawning, gym wrappers, contact sensors, and task-level behavior.
