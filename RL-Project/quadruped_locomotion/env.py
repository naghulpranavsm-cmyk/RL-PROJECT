"""Gymnasium MuJoCo environment with gamified rules and procedural terrain."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install dependencies with `pip install -e .` before using the environment.") from exc

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError("MuJoCo is required. Install dependencies with `pip install -e .`.") from exc

from quadruped_locomotion.controller import GaitController, LocomotionCommand
from quadruped_locomotion.mjcf import (
    FOOT_GEOM_NAMES,
    JOINT_NAMES,
    MAX_TORQUE,
    ROOT_BODY,
    ROOT_JOINT,
    REST_POSE,
    SPAWN_HEIGHT,
    build_world_xml,
)
from quadruped_locomotion.terrain import TerrainSpec, generate_terrain


MAX_HEALTH = 100.0
UPRIGHT_THRESHOLD = 0.35
STABLE_UPRIGHT_THRESHOLD = 0.72
#: Torso height above the local floor that counts as standing rather than folded.
STANDING_HEIGHT = 0.22


@dataclass
class RewardConfig:
    """Reward weights.

    The ``spec`` values are the literal table from the design brief. Because
    ``stability_per_step`` is paid every control step while ``forward_per_metre``
    is paid per metre, the literal table makes standing still worth roughly as
    much as walking. The ``balanced`` profile keeps every sign and every reward
    term but rescales the per-step terms so locomotion dominates, which is what
    an RL policy needs in order to actually learn to walk.
    """

    forward_per_metre: float = 1.0
    speed_bonus: float = 0.1
    stability_per_step: float = 0.5
    clearance_bonus: float = 3.0
    collision_penalty: float = 5.0
    fall_penalty: float = 10.0
    idle_penalty: float = 0.1
    backward_penalty: float = 2.0
    collision_damage: float = 8.0
    fall_damage: float = 20.0
    level_clear_bonus: float = 25.0
    idle_epsilon: float = 0.0008
    backward_epsilon: float = 0.0015
    collision_cooldown_steps: int = 25
    combo_window: int = 35
    combo_step: float = 0.10
    combo_cap: int = 10
    health: float = MAX_HEALTH

    @classmethod
    def for_profile(cls, profile: str) -> "RewardConfig":
        if profile == "spec":
            return cls()
        if profile == "balanced":
            return replace(
                cls(),
                stability_per_step=0.05,
                idle_penalty=0.02,
                collision_penalty=4.0,
                fall_penalty=8.0,
                backward_penalty=1.0,
                clearance_bonus=4.0,
                combo_window=40,
            )
        raise ValueError(f"Unknown reward profile {profile!r}; expected 'spec' or 'balanced'.")


@dataclass
class GameState:
    """Player-facing state that is also exposed to the observation vector."""

    score: float = 0.0
    best_score: float = 0.0
    health: float = MAX_HEALTH
    level: int = 1
    combo: int = 0
    combo_multiplier: float = 1.0
    falls: int = 0
    collisions: int = 0
    hazards_cleared: int = 0
    levels_cleared: int = 0


@dataclass
class _Tracking:
    """Internal counters that must not leak into observations."""

    previous_x: float = 0.0
    last_safe_x: float = 0.0
    downed: bool = False
    down_steps: int = 0
    collapse_steps: int = 0
    collision_cooldown: int = 0
    hazard_contact: bool = False
    jump_cooldown: int = 0
    jump_latched: bool = False
    jump_assist: float = 0.0
    elapsed_steps: int = 0
    cleared_hazards: set[str] = field(default_factory=set)


class QuadrupedGameEnv(gym.Env):
    """Torque-controlled quadruped locomotion task with game rules.

    ``step`` accepts either a raw 12D torque vector (used by RL policies) or a
    locomotion-command mapping (used by manual play), so both entry points share
    one physics, reward, and rules implementation.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        seed: int | None = None,
        frame_skip: int = 10,
        track_length: float = 80.0,
        start_level: int = 1,
        reward_profile: str = "spec",
        time_limit_s: float = 30.0,
        down_rescue_s: float = 0.6,
        collapse_rescue_s: float = 1.2,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = int(frame_skip)
        self.track_length = float(track_length)
        self.start_level = max(1, int(start_level))
        self.reward_config = RewardConfig.for_profile(reward_profile)
        self.reward_profile = reward_profile
        self.max_steps = max(1, int(round(time_limit_s * self.metadata["render_fps"])))
        self.down_rescue_steps = max(1, int(round(down_rescue_s * self.metadata["render_fps"])))
        self.collapse_rescue_steps = max(
            1, int(round(collapse_rescue_s * self.metadata["render_fps"]))
        )
        self.base_seed = seed

        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self._pending_reset = False

        # Policies emit a normalised torque fraction which is scaled to the
        # actuator limit. A raw +/-45 space makes PPO's gradient scale badly
        # across 12 joints, and obs clamping already assumes normalised actions.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = self._build_observation_space()

        self.controller = GaitController()
        self.game = GameState(level=self.start_level)
        self.tracking = _Tracking()
        self.last_action = np.zeros(12, dtype=np.float32)
        self.command = LocomotionCommand()

        self._renderer: mujoco.Renderer | None = None
        self._renderer_version = -1
        self._camera = mujoco.MjvCamera()
        self.model_version = 0

        self.terrain: TerrainSpec = generate_terrain(seed, self.start_level, self.track_length)
        self.model = mujoco.MjModel.from_xml_string(build_world_xml(self.terrain))
        self.data = mujoco.MjData(self.model)
        self._bind_indices()
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------

    def _build_observation_space(self) -> spaces.Box:
        max_joint = np.array(
            [0.95, 1.55, 0.90] * 4,
            dtype=np.float64,
        )
        low = np.concatenate(
            [
                np.full(3, -12.0),
                np.full(3, -40.0),
                np.full(4, -1.0),
                np.full(1, -1.5),
                -max_joint,
                np.full(12, -60.0),
                np.full(12, -MAX_TORQUE),
                np.full(2, -1.0),
                np.full(2, 0.0),
                np.full(3, -1.5),
                np.full(1, 0.0),
                np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
            ]
        )
        high = np.concatenate(
            [
                np.full(3, 12.0),
                np.full(3, 40.0),
                np.full(4, 1.0),
                np.full(1, 1.5),
                max_joint,
                np.full(12, 60.0),
                np.full(12, MAX_TORQUE),
                np.full(2, 1.0),
                np.full(2, 1.0),
                np.full(3, 1.5),
                np.full(1, 1.0),
                np.array([1.0, 1.0, 3.0, 1.0, 1.0]),
            ]
        )
        return spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32)

    def _bind_indices(self) -> None:
        model = self.model
        self.torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
        root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_JOINT)
        if root_joint < 0:
            raise RuntimeError("Root free joint is missing from the generated model.")

        self.root_qpos_adr = int(model.jnt_qposadr[root_joint])
        self.root_qvel_adr = int(model.jnt_dofadr[root_joint])
        self.root_x = slice(self.root_qpos_adr, self.root_qpos_adr + 3)
        self.root_quat = slice(self.root_qpos_adr + 3, self.root_qpos_adr + 7)
        self.root_linvel = slice(self.root_qvel_adr, self.root_qvel_adr + 3)
        self.root_angvel = slice(self.root_qvel_adr + 3, self.root_qvel_adr + 6)

        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        if any(jid < 0 for jid in joint_ids):
            missing = [n for n, jid in zip(JOINT_NAMES, joint_ids) if jid < 0]
            raise RuntimeError(f"Missing actuated joints: {missing}")
        self.joint_qpos_adr = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int64)
        self.joint_qvel_adr = np.array([model.jnt_dofadr[jid] for jid in joint_ids], dtype=np.int64)

        self.platform_qpos_adr: dict[str, int] = {}
        self.platform_qvel_adr: dict[str, int] = {}
        for platform in self.terrain.moving_platforms:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, platform.joint_name)
            if jid >= 0:
                self.platform_qpos_adr[platform.joint_name] = int(model.jnt_qposadr[jid])
                self.platform_qvel_adr[platform.joint_name] = int(model.jnt_dofadr[jid])

        self._hazard_geom_ids = frozenset(
            gid
            for hazard in self.terrain.hazards
            if (gid := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, hazard.name)) >= 0
        )
        self._foot_geom_ids = frozenset(
            gid
            for name in FOOT_GEOM_NAMES
            if (gid := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0
        )

    def _rebuild_track(self, *, level: int, seed: int | None) -> None:
        self.terrain = generate_terrain(seed, level, self.track_length)
        self.model = mujoco.MjModel.from_xml_string(build_world_xml(self.terrain))
        self.data = mujoco.MjData(self.model)
        self._bind_indices()
        self.model_version += 1
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._renderer_version = -1

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep * self.frame_skip)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)

        wants_new_track = bool(options and options.get("new_track", False))
        if wants_new_track and not self._pending_reset:
            level = int(options.get("level", self.game.level))
            self._rebuild_track(level=level, seed=self._seed)
            self.game = GameState(level=level, health=self.game.health)

        self._pending_reset = False
        self._reset_robot_pose()
        self.tracking = _Tracking(previous_x=self._robot_x, last_safe_x=self._robot_x)
        self.tracking.elapsed_steps = 0
        # A new episode starts from a clean slate. Health and the run counters
        # must be restored too: if they leak, an episode that reached zero health
        # leaves every later episode terminating on its very first step. Only
        # session records carry over (best score, current level, levels cleared).
        self.game.score = 0.0
        self.game.health = MAX_HEALTH
        self.game.combo = 0
        self.game.combo_multiplier = 1.0
        self.game.falls = 0
        self.game.collisions = 0
        self.game.hazards_cleared = 0
        self.last_action = np.zeros(12, dtype=np.float32)
        self.command = LocomotionCommand()
        self.controller.reset()
        return self._get_obs(), self.info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        command, raw_torques = self._resolve_action(action)

        if command is not None and command.reset:
            self._respawn(self.tracking.last_safe_x)

        # The impulse is applied before the controller runs so the leg-extension
        # assist that follows it lands on the same frame as the launch. The jump
        # is edge-triggered: holding the key must not re-launch the robot every
        # cooldown while it is still trying to stay upright on the way down.
        jump_edge = command is not None and command.jump and not self.tracking.jump_latched
        self.tracking.jump_latched = bool(command is not None and command.jump)
        if jump_edge and self.tracking.jump_cooldown <= 0 and self._can_jump():
            self.data.qvel[self.root_qvel_adr + 2] += 1.8
            self.tracking.jump_cooldown = int(round(0.5 / self.dt))
            self.tracking.jump_assist = 0.10
        self.tracking.jump_assist = max(0.0, self.tracking.jump_assist - self.dt)

        if command is not None:
            torques = self.controller.action(
                self.data.qpos[self.joint_qpos_adr],
                self.data.qvel[self.joint_qvel_adr],
                self.dt,
                command,
                jump_assist=self.tracking.jump_assist,
            )
        else:
            torques = raw_torques

        self.data.ctrl[:] = torques
        self.last_action = np.asarray(torques, dtype=np.float32)

        for _ in range(self.frame_skip):
            self._update_moving_platforms()
            mujoco.mj_step(self.model, self.data)

        self.tracking.jump_cooldown = max(0, self.tracking.jump_cooldown - 1)
        self.tracking.elapsed_steps += 1
        reward, events = self._compute_reward()
        self.game.score += reward
        self.game.best_score = max(self.game.best_score, self.game.score)

        level_cleared = self._robot_x >= self.terrain.x_max
        truncated = bool(level_cleared or self.tracking.elapsed_steps >= self.max_steps)
        terminated = bool(self.game.health <= 0.0)

        if level_cleared:
            reward += self.reward_config.level_clear_bonus
            self.game.score += self.reward_config.level_clear_bonus
            self.game.levels_cleared += 1
            self.game.level += 1
            events["level_cleared"] = True
            self._rebuild_track(level=self.game.level, seed=self._derive_level_seed())
            self._reset_robot_pose()
            self._pending_reset = True
            self.tracking.previous_x = self._robot_x
            self.tracking.last_safe_x = self._robot_x

        info = self.info(events)
        return self._get_obs(), float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._renderer_version = -1

    # ------------------------------------------------------------------
    # action handling
    # ------------------------------------------------------------------

    def _resolve_action(
        self, action: Any
    ) -> tuple[LocomotionCommand | None, np.ndarray]:
        """Split a manual command from a raw torque vector from the policy.

        Returns ``(None, torques)`` when the caller supplied a raw 12D action, so
        the controller and the jump assist stay out of the RL path.
        """

        if isinstance(action, dict):
            self.command = LocomotionCommand(
                forward=float(action.get("forward", 0.0)),
                turn=float(action.get("turn", 0.0)),
                jump=bool(action.get("jump", False)),
                sprint=bool(action.get("sprint", False)),
                reset=bool(action.get("reset", False)),
            )
            return self.command, np.zeros(12, dtype=np.float64)
        # A policy owns the joints directly, so the last manual command must not
        # linger in the observation.
        self.command = LocomotionCommand()
        normalised = np.asarray(action, dtype=np.float64).reshape(12)
        return None, np.clip(normalised, -1.0, 1.0) * MAX_TORQUE

    # ------------------------------------------------------------------
    # physics state helpers
    # ------------------------------------------------------------------

    @property
    def _robot_x(self) -> float:
        return float(self.data.qpos[self.root_x.start])

    @property
    def _robot_z(self) -> float:
        return float(self.data.qpos[self.root_x.start + 2])

    def _uprightness(self) -> float:
        return float(self.data.xmat[self.torso_id, 8])

    def _reset_robot_pose(self, x: float | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self._pose_robot(x if x is not None else 0.0)
        mujoco.mj_forward(self.model, self.data)

    def _pose_robot(self, x: float) -> None:
        base = self.root_qpos_adr
        self.data.qpos[base + 0] = x
        self.data.qpos[base + 1] = 0.0
        self.data.qpos[base + 2] = SPAWN_HEIGHT
        self.data.qpos[base + 3] = 1.0
        self.data.qpos[base + 4 : base + 7] = 0.0
        self.data.qvel[self.root_qvel_adr : self.root_qvel_adr + 6] = 0.0
        rest = REST_POSE
        for i, name in enumerate(JOINT_NAMES):
            self.data.qpos[self.joint_qpos_adr[i]] = rest.get(name.split("_", 1)[1], 0.0)

    def _respawn(self, x: float) -> None:
        safe = self._nearest_safe_x(x)
        self._pose_robot(safe)
        mujoco.mj_forward(self.model, self.data)
        self.tracking.previous_x = self._robot_x
        self.tracking.last_safe_x = safe
        self.tracking.downed = False
        self.tracking.down_steps = 0
        self.tracking.collapse_steps = 0
        self.tracking.jump_cooldown = 0
        self.tracking.jump_latched = False
        self.tracking.jump_assist = 0.0
        self.controller.reset()

    def _nearest_safe_x(self, x: float) -> float:
        lo = float(self.terrain.x_min + 0.5)
        hi = float(self.terrain.x_max - 0.5)
        candidate = float(np.clip(x, lo, hi))
        for delta in np.arange(0.0, 4.0, 0.25):
            for sign in (1.0, -1.0):
                probe = candidate + sign * float(delta)
                if probe < lo or probe > hi:
                    continue
                if not self.terrain.has_ground(probe):
                    continue
                if any(abs(probe - h.x) < 0.9 for h in self.terrain.hazards):
                    continue
                return probe
        return lo

    def _update_moving_platforms(self) -> None:
        t = float(self.data.time)
        level_speed = 1.0 + 0.12 * max(0, self.game.level - 1)
        for platform in self.terrain.moving_platforms:
            qadr = self.platform_qpos_adr.get(platform.joint_name)
            if qadr is None:
                continue
            vadr = self.platform_qvel_adr.get(platform.joint_name)
            omega = platform.speed * level_speed
            theta = platform.phase + t * omega
            self.data.qpos[qadr] = platform.amplitude * float(np.sin(theta))
            if vadr is not None:
                self.data.qvel[vadr] = platform.amplitude * omega * float(np.cos(theta))

    def _scan_contacts(self) -> tuple[bool, bool]:
        hazard_hit = False
        grounded = False
        contacts = self.data.contact
        hazards = self._hazard_geom_ids
        feet = self._foot_geom_ids
        for i in range(self.data.ncon):
            g1 = int(contacts[i].geom1)
            g2 = int(contacts[i].geom2)
            if g1 in hazards or g2 in hazards:
                hazard_hit = True
            if g1 in feet or g2 in feet:
                grounded = True
        return hazard_hit, grounded

    def _is_grounded(self) -> bool:
        return self._scan_contacts()[1]

    def _can_jump(self) -> bool:
        """Allow a launch when the robot is on the floor, not only when a foot
        geom happens to be in contact.

        After a spawn or respawn the feet take a frame or two to settle, so a
        strict contact test would swallow the very first jump press.
        """

        if self._is_grounded():
            return True
        ground = self.terrain.ground_height(self._robot_x)
        base = 0.0 if ground is None else ground
        vertical_speed = float(self.data.qvel[self.root_qvel_adr + 2])
        # SPAWN_HEIGHT is the resting torso height, so this is "still at rest
        # level", not a height above the floor.
        return abs(vertical_speed) < 0.35 and self._robot_z < base + SPAWN_HEIGHT + 0.06

    # ------------------------------------------------------------------
    # reward
    # ------------------------------------------------------------------

    def _compute_reward(self) -> tuple[float, dict[str, Any]]:
        cfg = self.reward_config
        x = self._robot_x
        dx = x - self.tracking.previous_x
        vx = float(self.data.qvel[self.root_qvel_adr])
        upright = self._uprightness()
        hazard_hit, grounded = self._scan_contacts()

        reference_ground = self.terrain.ground_height(x)
        base_ground = 0.0 if reference_ground is None else reference_ground
        over_gap = reference_ground is None
        stable = bool(
            upright > STABLE_UPRIGHT_THRESHOLD
            and grounded
            and not over_gap
            and self._robot_z > base_ground + STANDING_HEIGHT
        )
        # Over a gap there is no floor to compare against, so treat "sitting well
        # below the standing torso height" as having dropped in. Without this a
        # robot that wedges a leg in a narrow hole stays upright, scores no
        # stability, but is never rescued either.
        fell = bool(
            upright < UPRIGHT_THRESHOLD
            or self._robot_z < base_ground - 0.30
            or (over_gap and self._robot_z < base_ground + 0.28)
        )
        # A robot that folds its legs but stays level never trips the checks
        # above, yet it cannot walk and would sit there scoring nothing. Count it
        # as a knockdown once it has clearly stopped trying to stand.
        folded = bool(
            not over_gap
            and upright >= UPRIGHT_THRESHOLD
            and self._robot_z < base_ground + STANDING_HEIGHT
        )
        self.tracking.collapse_steps = self.tracking.collapse_steps + 1 if folded else 0
        if self.tracking.collapse_steps > self.collapse_rescue_steps:
            fell = True

        reward = cfg.forward_per_metre * dx
        reward += cfg.speed_bonus * vx
        if stable:
            reward += cfg.stability_per_step
        if abs(dx) < cfg.idle_epsilon:
            reward -= cfg.idle_penalty
        if dx < -cfg.backward_epsilon:
            reward -= cfg.backward_penalty

        # Damage on the frame contact begins, not for every frame spent leaning
        # on a wall; the cooldown then only guards a genuinely new impact.
        impact = bool(hazard_hit and not self.tracking.hazard_contact)
        self.tracking.hazard_contact = hazard_hit
        collision = bool(impact and self.tracking.collision_cooldown <= 0)
        if collision:
            reward -= cfg.collision_penalty
            self.game.health -= cfg.collision_damage
            self.game.combo = 0
            self.game.collisions += 1
            self.tracking.collision_cooldown = cfg.collision_cooldown_steps
        # The cooldown must tick down whether or not a hazard is in contact,
        # otherwise it stalls at a positive value and the next graze is ignored.
        self.tracking.collision_cooldown = max(0, self.tracking.collision_cooldown - 1)

        if fell and not self.tracking.downed:
            reward -= cfg.fall_penalty
            self.game.health -= cfg.fall_damage
            self.game.combo = 0
            self.game.falls += 1
            self.tracking.downed = True
        elif not fell:
            self.tracking.downed = False
        self.tracking.down_steps = self.tracking.down_steps + 1 if fell else 0

        clearance_bonus = self._obstacle_clearance_bonus(x)
        reward += clearance_bonus

        progressing = dx > cfg.idle_epsilon and stable and not collision and not fell
        if progressing:
            self.game.combo += 1
        elif not collision and not fell:
            self.game.combo = max(0, self.game.combo - 1)
        self.game.combo_multiplier = 1.0 + min(
            self.game.combo // max(1, cfg.combo_window), cfg.combo_cap
        ) * cfg.combo_step
        if reward > 0.0:
            reward *= self.game.combo_multiplier

        if stable and grounded and upright > 0.85 and not over_gap:
            self.tracking.last_safe_x = x

        self.tracking.previous_x = x

        out_of_world = self._robot_z < base_ground - 0.55
        # Give a tipped or wedged robot half a second to recover on its own, then
        # rescue it to the last patch of solid ground. A gap has no floor to fall
        # below, so the knockdown timer is what ends a fall into one.
        if out_of_world or self.tracking.down_steps > self.down_rescue_steps:
            self._respawn(self.tracking.last_safe_x)

        events = {
            "fell": bool(fell),
            "stable": bool(stable),
            "grounded": bool(grounded),
            "obstacle_collision": bool(collision),
            "clearance_bonus": float(clearance_bonus),
            "forward_progress": float(dx),
            "velocity": float(vx),
            "uprightness": float(upright),
            "combo_progress": bool(progressing),
            "level_cleared": False,
        }
        return float(reward), events

    def _obstacle_clearance_bonus(self, x: float) -> float:
        bonus = 0.0
        cleared = self.tracking.cleared_hazards
        for hazard in self.terrain.hazards:
            if hazard.name not in cleared and x > hazard.clearance_x:
                cleared.add(hazard.name)
                bonus += self.reward_config.clearance_bonus
                self.game.hazards_cleared += 1
        return bonus

    def _derive_level_seed(self) -> int | None:
        if self._seed is None:
            return None
        return int(self._seed) + 1000 * self.game.level

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        x = self._robot_x
        quat = self.data.qpos[self.root_quat]
        here = self.terrain.ground_height(x)
        base = 0.0 if here is None else here

        probes = []
        for offset in (0.45, 0.95, 1.60):
            h = self.terrain.ground_height(x + offset)
            probes.append(0.0 if h is None else float(h - base))
        gap_ahead = 1.0 if self.terrain.is_gap(x + 0.95) else 0.0

        health_norm = float(np.clip(self.game.health / MAX_HEALTH, 0.0, 1.0))
        level_norm = float(np.clip(self.game.level / 10.0, 0.0, 1.0))
        progress = float(np.clip(x / max(1.0, self.terrain.x_max), 0.0, 1.0))

        obs = np.concatenate(
            [
                np.asarray(self.data.qvel[self.root_linvel], dtype=np.float64),
                np.asarray(self.data.qvel[self.root_angvel], dtype=np.float64),
                np.asarray(quat, dtype=np.float64),
                np.array([self._robot_z - base], dtype=np.float64),
                self.data.qpos[self.joint_qpos_adr].astype(np.float64),
                self.data.qvel[self.joint_qvel_adr].astype(np.float64),
                self.last_action.astype(np.float64),
                np.array(
                    [
                        float(np.clip(self.command.forward, -1.0, 1.0)),
                        float(np.clip(self.command.turn, -1.0, 1.0)),
                        1.0 if self.command.jump else 0.0,
                        1.0 if self.command.sprint else 0.0,
                    ]
                ),
                np.asarray(probes, dtype=np.float64),
                np.array([gap_ahead], dtype=np.float64),
                np.array(
                    [
                        health_norm,
                        level_norm,
                        float(np.clip(self.game.combo_multiplier, 1.0, 3.0)),
                        progress,
                        self._uprightness(),
                    ]
                ),
            ]
        )
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        low = self.observation_space.low
        high = self.observation_space.high
        return np.clip(obs, low, high).astype(np.float32)

    def info(self, events: dict[str, Any] | None = None) -> dict[str, Any]:
        info: dict[str, Any] = {
            "score": float(self.game.score),
            "best_score": float(self.game.best_score),
            "health": float(self.game.health),
            "level": int(self.game.level),
            "combo": int(self.game.combo),
            "combo_multiplier": float(self.game.combo_multiplier),
            "x_position": float(self._robot_x),
            "track_length": float(self.terrain.length),
            "falls": int(self.game.falls),
            "collisions": int(self.game.collisions),
            "hazards_cleared": int(self.game.hazards_cleared),
            "levels_cleared": int(self.game.levels_cleared),
            "model_version": int(self.model_version),
        }
        if events:
            info.update(events)
        return info

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        return self.render_rgb()

    def render_rgb(self, width: int = 1280, height: int = 720) -> np.ndarray:
        if (
            self._renderer is None
            or self._renderer_version != self.model_version
            or self._renderer.width != width
            or self._renderer.height != height
        ):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderer_version = self.model_version

        self.follow_camera(self._camera)
        self._renderer.update_scene(self.data, camera=self._camera)
        return self._renderer.render()

    def follow_camera(self, camera: mujoco.MjvCamera, distance: float = 4.2) -> None:
        """Aim a free camera down the track from just behind and above the robot."""
        torso = self.data.xpos[self.torso_id]
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (float(torso[0]) + 0.80, float(torso[1]), max(0.45, float(torso[2])))
        camera.distance = distance
        camera.azimuth = 105.0
        camera.elevation = -18.0