"""Command-to-torque gait controller used by manual play."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quadruped_locomotion.mjcf import JOINT_NAMES, REST_POSE


LEG_ORDER = ("fl", "fr", "rl", "rr")
FRONT_LEGS = ("fl", "fr")
LEFT_LEGS = ("fl", "rl")
#: Trot pairing: diagonal legs move together.
PHASE_OFFSET = {"fl": 0.0, "rr": 0.0, "fr": np.pi, "rl": np.pi}


@dataclass
class LocomotionCommand:
    """A high-level command, mapped from the keyboard during manual play."""

    forward: float = 0.0
    turn: float = 0.0
    jump: bool = False
    sprint: bool = False
    reset: bool = False


class GaitController:
    """Turns a locomotion command into joint torques.

    The gait is a diagonal trot built from position targets and a PD law, which
    gives manual play an immediate, responsive feel while leaving RL policies
    free to learn direct torque control on the same model.
    """

    def __init__(self, joint_names: tuple[str, ...] = JOINT_NAMES) -> None:
        self.joint_names = joint_names
        self.phase = 0.0
        self.kp = np.array([30.0, 26.0, 14.0] * 4, dtype=np.float64)
        self.kd = np.array([2.0, 1.5, 0.9] * 4, dtype=np.float64)
        self.base_cadence = 18.0
        self.cadence_gain = 0.0
        self.sprint_cadence = 8.0
        self.sprint_speed_scale = 1.15
        self.hip_amplitude = 0.60
        self.hip_compensation = 0.0
        self.lift_amplitude = 1.20
        self.turn_stride_gain = 1.40
        self.turn_speed_relief = 0.45
        self.turn_inside_scale = 0.0
        self.max_stride = 1.40
        # Joint-range headroom: hip stops at 0.95, and the ankle starts at
        # REST_POSE["ankle"] and stops at 1.60, so 1.2 of lift is the ceiling.
        self.max_hip = 0.90
        self.max_lift = 1.20
        self._targets = np.zeros(12, dtype=np.float64)

    def reset(self) -> None:
        self.phase = 0.0

    def action(
        self,
        qpos_joints: np.ndarray,
        qvel_joints: np.ndarray,
        dt: float,
        command: LocomotionCommand,
        jump_assist: float = 0.0,
    ) -> np.ndarray:
        speed = float(np.clip(command.forward, -1.0, 1.0))
        turn = float(np.clip(command.turn, -1.0, 1.0))
        # Sprint raises cadence and stride rather than torque. Scaling torque past
        # what the feet can grip just scrabbles the gait and slows the robot down.
        speed *= self.sprint_speed_scale if command.sprint else 1.0

        drive = min(1.0, abs(speed) + abs(turn))
        if drive < 1e-3 and jump_assist <= 0.0:
            self._targets[:] = [REST_POSE["hip"], REST_POSE["knee"], REST_POSE["ankle"]] * 4
            return np.clip(
                self.kp * (self._targets - qpos_joints) - self.kd * qvel_joints,
                -45.0,
                45.0,
            )

        cadence = (
            self.base_cadence
            + self.cadence_gain * drive
            + (self.sprint_cadence if command.sprint else 0.0)
        ) * drive
        self.phase = (self.phase + dt * cadence) % (2.0 * np.pi)

        for leg_index, leg in enumerate(LEG_ORDER):
            leg_phase = self.phase + PHASE_OFFSET[leg]

            # Stance occupies phase [0, pi): the foot sweeps from front to back
            # while planted, which is what pushes the body forward. Swing
            # occupies [pi, 2*pi): the foot is lifted and returns to the front.
            sweep = float(np.cos(leg_phase))
            lift = max(0.0, -float(np.sin(leg_phase)))

            side = 1.0 if leg in LEFT_LEGS else -1.0
            # Turning shortens the stride on the inside legs and lengthens it on
            # the outside legs, because every joint is a pitch hinge and yaw can
            # only come from asymmetric stepping.
            # Turning shortens the stride on the inside legs and lengthens it on
            # the outside legs, because every joint is a pitch hinge and yaw can
            # only come from asymmetric stepping. Backing the authority off as
            # forward speed builds keeps a running turn inside what the hips can
            # actually deliver; at full speed a standing turn's spread tips the
            # robot over.
            turn_authority = 1.0 - self.turn_speed_relief * min(1.0, abs(speed))
            stride = speed - turn * side * self.turn_stride_gain * turn_authority
            # Without a clamp the outside leg of a turn can be commanded past a
            # full stride, which the hip torque can no longer track and which
            # tips the robot over under sprint.
            stride = float(np.clip(stride, -self.max_stride, self.max_stride))
            # Steering on the spot: the inside leg takes a reduced counter-stride.
            # Letting it swing at full amplitude scrubs the feet and slowly drags
            # the body over, but planting it completely removes the yaw couple.
            if abs(speed) < 0.05 and stride * turn < 0.0:
                stride *= self.turn_inside_scale
            hip_amp = self.hip_amplitude * stride
            lift_amp = self.lift_amplitude * min(1.0, abs(stride))
            # Keep the commanded targets inside the joint ranges. Ankle and knee
            # are the binding pair here: the rest pose already sits at +0.35, so
            # a lift above ~1.2 drives the ankle past its stop and the reference
            # clips, which silently wrecks the whole gait.
            hip_amp = float(np.clip(hip_amp, -self.max_hip, self.max_hip))
            lift_amp = min(lift_amp, self.max_lift)

            hip_target = -hip_amp * sweep
            # Stance legs keep the rest length so they hold the body up; only the
            # swing leg shortens. The ankle cancels that shortening so the foot
            # link stays level while it is off the ground.
            knee_target = REST_POSE["knee"] - self.hip_compensation * hip_target - lift_amp * lift
            ankle_target = REST_POSE["ankle"] - self.hip_compensation * hip_target + lift_amp * lift

            base = leg_index * 3
            self._targets[base : base + 3] = (hip_target, knee_target, ankle_target)

        torques = self.kp * (self._targets - qpos_joints) - self.kd * qvel_joints
        if jump_assist > 0.0:
            torques[1::3] -= 26.0
            torques[2::3] -= 16.0
        return np.clip(torques, -45.0, 45.0)