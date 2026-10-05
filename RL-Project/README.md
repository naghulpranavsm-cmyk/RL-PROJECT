# Gamified MuJoCo Quadruped Locomotion

A Gymnasium-compatible MuJoCo environment where a 12-joint quadruped learns to walk, sprint, and jump across a procedurally generated obstacle course, with a score/health/level game loop and a keyboard play mode.

## Features

- 12 torque-actuated joints: hip, knee, and ankle for each of the four legs, limited to +/-45 Nm.
- MuJoCo physics with `gravity = -9.81`, `timestep = 0.002`, `frame_skip = 10` (50 Hz control), frictional contacts, and torque control.
- Procedural terrain with ramps, uneven tiles, gaps, slippery zones, walls, spike hazards, and laterally sliding platforms. Difficulty scales with level.
- Reward computed on every `env.step()`:
  - Forward progress: `+1` per unit distance.
  - Speed bonus: `+0.1 * velocity`.
  - Stability: `+0.5` per upright control step.
  - Obstacle clearance bonus.
  - Obstacle collision: `-5`.
  - Falling: `-10`.
  - Idle behaviour: `-0.1`.
  - Moving backward: `-2`.
- Game loop: score, health, level progression on reaching the end of a track, combo multiplier, and episode termination at zero health.
- Automatic rescue: a robot that tips over, drops into a gap, or folds its legs and cannot stand is returned to the last solid ground after a short grace period.
- Manual keyboard mode with a dynamic follow camera and HUD overlay.
- RL training entry point using Stable-Baselines3 PPO.

### Reward profiles

`RewardConfig` has two profiles, selected with `--reward-profile`:

- `spec` (default for manual play) implements the reward table above literally.
- `balanced` damps the per-step terms (`stability_per_step`, `idle_penalty`, and the penalties) and is the **default for training**.

The reason is arithmetic, not preference: at 50 Hz the literal `+0.5` stability term pays 25 per second just for standing still, which dwarfs the `+1` per metre of forward progress. A policy trained on the literal profile is rewarded for doing nothing. Manual play still uses `spec` so the HUD shows the numbers from the brief.

## Install

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[play,train,dev]"
```

- `play` pulls in `glfw`, which the manual play script needs for its own window.
- `train` pulls in Stable-Baselines3 and PyTorch.
- `dev` pulls in pytest.

MuJoCo rendering needs a working OpenGL context. On a headless machine, train without the play script.

## Manual Play

```bash
python scripts/manual_play.py
```

Useful flags: `--level N`, `--track-length M`, `--reward-profile {spec,balanced}`, `--seed N`.

Controls:

- `W` / Up: forward
- `S` / Down: backward
- `A` / Left: turn left
- `D` / Right: turn right
- `Space`: jump (one launch per press)
- `Shift`: sprint
- `R`: reset the robot to the last safe position
- `N`: generate a new track
- `P`: pause
- `Esc`: quit

The HUD shows level, score, a health bar, combo multiplier, per-step reward, distance, falls, and collisions.

### About the scripted gait

`GaitController` is a hand-tuned diagonal trot used only for manual play, so the robot is drivable without training anything. It is deliberately conservative: amplitudes are clamped to the joint ranges and it will not attempt anything that topples the robot. Measured over 700 control steps on flat ground, it walks about 4 m forward, sprints about 4.3 m, and reverses about 4.5 m without falling.

Steering is weaker than translation: turning in place produces roughly 50-70 degrees of heading change over 10 seconds, so steering while moving (`W` + `A`) is the more responsive way to change direction. Learning a proper gait is what the RL path is for.

## Train

```bash
python scripts/train_ppo.py --timesteps 200000 --model-out runs/ppo_quadruped
```

Useful flags: `--reward-profile {balanced,spec}`, `--n-envs N`, `--start-level N`, `--track-length M`, `--learning-rate LR`, `--device {auto,cpu,cuda}`, `--seed N`.

`--n-envs` above 1 automatically uses SB3's subprocess vectorized env, so each env gets its own core.

Checkpoints land in `runs/checkpoints/`, the best-eval policy in `runs/best/`, and TensorBoard logs in `runs/tensorboard/`. The final policy is written to `--model-out`.

The observation is a 60D vector: root linear and angular velocity, root orientation quaternion, torso height above the local floor, all 12 joint positions and velocities, the previous torque, the current command, three terrain probes, a gap-ahead flag, and health/level/combo/progress/uprightness.

The action space is a 12D `Box(-1, 1)`. Each entry is a fraction of the actuator limit and is scaled to +/-45 Nm internally. A normalised action space is used because a raw newton-metre space makes PPO's gradient scale badly across 12 joints; `data.ctrl` still holds real torque.

`env.step()` accepts either the raw 12D torque vector (used by policies) or a command dictionary such as `{"forward": 1.0, "turn": 0.0, "jump": False, "sprint": False, "reset": False}` (used by manual play). Supplying a raw action clears any manual command so a policy never inherits stale command state in its observations.

## Project Layout

```text
quadruped_locomotion/
  env.py          # Gymnasium environment, reward, game state, rendering
  mjcf.py         # Robot and world XML generation
  terrain.py      # Procedural terrain, hazards, platforms, analytic height queries
  controller.py   # Manual command-to-torque gait controller
scripts/
  manual_play.py  # Interactive keyboard mode with HUD
  train_ppo.py    # PPO training entry point
tests/
  test_environment.py
```

## Tests

```bash
python -m pytest tests -q
```

The suite covers the actuator count and space shapes, root-freejoint ordering, action-space scaling, per-seed determinism, score reset semantics, gap height queries, knockdown rescue, edge-triggered jumping, gait targets staying inside joint limits, forward/backward/steering behaviour, level clearing and model rebuilds, and health-based termination.
