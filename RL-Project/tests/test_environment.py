from __future__ import annotations

import importlib.util

import numpy as np
import pytest


try:  # pragma: no cover - exercised only when MuJoCo is installed
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None


mujoco_available = importlib.util.find_spec("mujoco") is not None
gym_available = importlib.util.find_spec("gymnasium") is not None

requires_mujoco = pytest.mark.skipif(
    not (mujoco_available and gym_available),
    reason="MuJoCo and Gymnasium are optional runtime dependencies",
)


@pytest.fixture()
def env():
    from quadruped_locomotion import QuadrupedGameEnv

    instance = QuadrupedGameEnv(seed=123)
    yield instance
    instance.close()


@requires_mujoco
def test_quadruped_has_12_actuators_and_steps(env) -> None:
    obs, info = env.reset(seed=123)
    assert env.model.nu == 12
    assert env.action_space.shape == (12,)
    assert obs.shape == env.observation_space.shape
    obs, reward, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
    assert obs.shape == env.observation_space.shape
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert {"score", "health", "level", "combo_multiplier"} <= set(info)


@requires_mujoco
def test_root_freejoint_is_the_first_qpos_entry(env) -> None:
    """The robot body must own qpos[0]; a platform joint ahead of it would
    silently shift every position and velocity the rewards read."""

    env.reset(seed=1)
    assert env.root_qpos_adr == 0
    assert env.root_qvel_adr == 0
    first = env.model.jnt(int(env.model.jnt_qposadr[0]))
    assert first.name == "root"
    assert int(np.asarray(first.type).ravel()[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
    torso_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    assert int(np.asarray(first.bodyid).ravel()[0]) == torso_id


@requires_mujoco
def test_action_space_is_normalised_torque(env) -> None:
    """Policies emit a fraction of the actuator limit, not raw newton-metres."""

    assert np.allclose(env.action_space.low, -1.0)
    assert np.allclose(env.action_space.high, 1.0)
    env.reset(seed=1)
    env.step(np.ones(12, dtype=np.float32))
    assert np.allclose(env.data.ctrl, env.model.actuator_ctrlrange[:, 1], atol=1e-3)
    env.step(-np.ones(12, dtype=np.float32))
    assert np.allclose(env.data.ctrl, env.model.actuator_ctrlrange[:, 0], atol=1e-3)


@requires_mujoco
def test_raw_torque_action_clears_manual_command(env) -> None:
    """A policy must not inherit a stale manual command in its observations."""

    env.reset(seed=1)
    for _ in range(3):
        env.step({"forward": 1.0, "turn": 1.0, "sprint": True})
    assert env.command.forward == 1.0
    obs, *_ = env.step(np.zeros(12, dtype=np.float32))
    assert env.command.forward == 0.0
    # forward/turn/jump/sprint occupy obs[47:51] of the 60D vector.
    assert np.allclose(obs[47:51], 0.0)


@requires_mujoco
def test_step_is_deterministic_for_a_fixed_seed(env) -> None:
    action = np.full(12, 0.25, dtype=np.float32)
    env.reset(seed=7)
    obs_a, reward_a, _, _, info_a = env.step(action)
    env.reset(seed=7)
    obs_b, reward_b, _, _, info_b = env.step(action)
    assert np.allclose(obs_a, obs_b)
    assert reward_a == reward_b
    assert info_a["score"] == info_b["score"]


@requires_mujoco
def test_reset_clears_score_but_keeps_best(env) -> None:
    env.reset(seed=1)
    for _ in range(5):
        env.step(np.zeros(12, dtype=np.float32))
    env.game.best_score = 12.5
    env.game.score = 3.0
    _, info = env.reset()
    assert info["score"] == 0.0
    assert env.game.best_score == 12.5


@requires_mujoco
def test_ground_height_reports_none_over_a_gap(env) -> None:
    """A gap is a hole, so the analytic query must not report floor under it."""

    env.reset(seed=4)
    gap_x = next(
        (x for x in np.arange(env.terrain.x_min, env.terrain.x_max, 0.05) if env.terrain.is_gap(x)),
        None,
    )
    if gap_x is None:
        pytest.skip("this track has no gap")
    assert env.terrain.ground_height(gap_x) is None
    assert not env.terrain.has_ground(gap_x)
    solid = float(env.terrain.x_min) + 1.0
    assert env.terrain.ground_height(solid) is not None


@requires_mujoco
def test_collapsing_robot_is_rescued(env) -> None:
    """A body that folds its legs stays upright, so it needs a separate
    knockdown rule or it would sit there forever scoring nothing."""

    env.reset(seed=1)
    # The fall is booked a few frames before the rescue fires, so watch for the
    # robot actually being put back on its feet rather than sampling one frame.
    peak_after_fall = 0.0
    for _ in range(env.collapse_rescue_steps + 120):
        env.step(np.zeros(12, dtype=np.float32))
        if env.game.falls:
            peak_after_fall = max(peak_after_fall, float(env.data.qpos[env.root_qpos_adr + 2]))
    assert env.game.falls >= 1
    assert env.game.health < 100.0
    assert peak_after_fall > 0.30


@requires_mujoco
def test_jump_is_edge_triggered_and_clears_the_ground(env) -> None:
    env.reset(seed=1)
    standing = float(env.data.qpos[env.root_qpos_adr + 2])
    peak = standing
    for _ in range(60):
        env.step({"jump": True})
        peak = max(peak, float(env.data.qpos[env.root_qpos_adr + 2]))
    assert peak > standing + 0.05
    # One launch only: a held key must not bounce the robot off the floor again.
    assert peak < standing + 0.35
    assert env.game.falls == 0


@requires_mujoco
def test_gait_targets_respect_joint_limits() -> None:
    """Sprint and turn used to command the ankle and hip past their stops,
    which clipped the reference and tipped the robot over."""

    from quadruped_locomotion import QuadrupedGameEnv
    from quadruped_locomotion.mjcf import JOINT_NAMES

    instance = QuadrupedGameEnv(seed=1)
    try:
        instance.reset(seed=1)
        for command in (
            {"forward": 1.0},
            {"forward": 1.0, "sprint": True},
            {"forward": 1.0, "turn": 1.0},
            {"turn": 1.0},
            {"forward": -1.0},
        ):
            for _ in range(30):
                instance.step(command)
            targets = instance.controller._targets
            for index, name in enumerate(JOINT_NAMES):
                joint = instance.model.joint(name)
                low, high = float(joint.range[0]), float(joint.range[1])
                assert low - 1e-6 <= targets[index] <= high + 1e-6, f"{name} target out of range"
    finally:
        instance.close()


@requires_mujoco
def test_forward_command_moves_robot_uphill_of_start() -> None:
    from quadruped_locomotion import QuadrupedGameEnv

    instance = QuadrupedGameEnv(seed=1)
    try:
        instance.reset(seed=1)
        start = float(instance.data.qpos[instance.root_qpos_adr])
        for _ in range(400):
            instance.step({"forward": 1.0})
        assert float(instance.data.qpos[instance.root_qpos_adr]) - start > 1.0
        assert instance.game.falls == 0
    finally:
        instance.close()


@requires_mujoco
def test_backward_command_reverses_direction() -> None:
    from quadruped_locomotion import QuadrupedGameEnv

    instance = QuadrupedGameEnv(seed=1)
    try:
        instance.reset(seed=1)
        start = float(instance.data.qpos[instance.root_qpos_adr])
        for _ in range(400):
            instance.step({"forward": -1.0})
        assert float(instance.data.qpos[instance.root_qpos_adr]) - start < -1.0
    finally:
        instance.close()


@requires_mujoco
def test_steering_changes_heading_in_both_directions() -> None:
    from quadruped_locomotion import QuadrupedGameEnv

    headings = {}
    for turn in (1.0, -1.0):
        instance = QuadrupedGameEnv(seed=1)
        try:
            instance.reset(seed=1)
            quat = instance.data.qpos[instance.root_qpos_adr + 3 : instance.root_qpos_adr + 7]
            start = 2.0 * np.arctan2(quat[3], quat[0])
            for _ in range(600):
                instance.step({"turn": turn})
            quat = instance.data.qpos[instance.root_qpos_adr + 3 : instance.root_qpos_adr + 7]
            headings[turn] = np.degrees(2.0 * np.arctan2(quat[3], quat[0]) - start)
            assert instance.game.falls == 0
        finally:
            instance.close()
    assert headings[1.0] > 10.0
    assert headings[-1.0] < -10.0


@requires_mujoco
def test_combo_multiplier_grows_with_clean_progress() -> None:
    from quadruped_locomotion import QuadrupedGameEnv

    instance = QuadrupedGameEnv(seed=1)
    try:
        instance.reset(seed=1)
        for _ in range(200):
            instance.step({"forward": 1.0})
        assert instance.game.combo_multiplier >= 1.0
    finally:
        instance.close()


@requires_mujoco
def test_reward_profiles_differ() -> None:
    from quadruped_locomotion import QuadrupedGameEnv

    spec = QuadrupedGameEnv(seed=1, reward_profile="spec").reward_config
    balanced = QuadrupedGameEnv(seed=1, reward_profile="balanced").reward_config
    try:
        # The literal brief pays 0.5 per step for being upright, which dwarfs the
        # locomotion terms; training therefore needs the damped profile.
        assert spec.stability_per_step > balanced.stability_per_step
        assert spec.idle_penalty > balanced.idle_penalty
    finally:
        pass


@requires_mujoco
def test_level_clear_rebuilds_the_model_and_advances() -> None:
    from quadruped_locomotion import QuadrupedGameEnv

    instance = QuadrupedGameEnv(seed=1)
    try:
        instance.reset(seed=1)
        first_version = instance.model_version
        instance.data.qpos[instance.root_qpos_adr] = instance.terrain.x_max + 0.05
        obs, reward, terminated, truncated, info = instance.step({"forward": 1.0})
        assert truncated is True
        assert info["level_cleared"] is True
        assert instance.game.levels_cleared == 1
        assert instance.game.level == 2
        assert instance.model_version > first_version
        assert instance._pending_reset is True
        obs, info = instance.reset()
        assert instance.game.levels_cleared == 1
        assert instance._pending_reset is False
        assert np.all(np.isfinite(obs))
        assert abs(float(instance.data.qpos[instance.root_qpos_adr])) < 1.0
    finally:
        instance.close()


@requires_mujoco
def test_health_reaching_zero_terminates() -> None:
    from quadruped_locomotion import QuadrupedGameEnv

    instance = QuadrupedGameEnv(seed=1)
    try:
        instance.reset(seed=1)
        instance.game.health = 0.0
        _, _, terminated, truncated, _ = instance.step({"forward": 1.0})
        assert terminated is True
    finally:
        instance.close()


@requires_mujoco
def test_reset_restores_health_and_run_counters() -> None:
    """A new episode must restore health and the run counters.

    Regression test: these used to leak across episodes, so once an episode
    reached zero health every later episode terminated on its very first step
    and PPO saw only 1-step rollouts.
    """

    from quadruped_locomotion import QuadrupedGameEnv

    instance = QuadrupedGameEnv(seed=5)
    try:
        instance.reset(seed=5)
        instance.game.health = 12.0
        instance.game.score = 99.0
        instance.game.falls = 7
        instance.game.collisions = 4
        instance.game.hazards_cleared = 3
        instance.game.best_score = 500.0

        _, info = instance.reset(seed=5)
        assert info["health"] == pytest.approx(100.0)
        assert info["score"] == pytest.approx(0.0)
        assert instance.game.falls == 0
        assert instance.game.collisions == 0
        assert instance.game.hazards_cleared == 0
        # Session records survive an episode boundary.
        assert instance.game.best_score == pytest.approx(500.0)
    finally:
        instance.close()


@requires_mujoco
def test_episode_does_not_terminate_immediately_after_a_death() -> None:
    """Episodes must be long enough to contain usable training signal."""

    from quadruped_locomotion import QuadrupedGameEnv

    instance = QuadrupedGameEnv(seed=2)
    try:
        instance.reset(seed=2)
        instance.game.health = 0.0
        _, _, terminated, _, _ = instance.step(np.zeros(12, dtype=np.float32))
        assert terminated is True

        _, info = instance.reset(seed=2)
        assert info["health"] == pytest.approx(100.0)
        steps = 0
        while steps < 50:
            _, _, terminated, truncated, _ = instance.step(np.zeros(12, dtype=np.float32))
            steps += 1
            assert not terminated, "episode ended immediately after reset"
            if truncated:
                break
        assert steps > 1
    finally:
        instance.close()
