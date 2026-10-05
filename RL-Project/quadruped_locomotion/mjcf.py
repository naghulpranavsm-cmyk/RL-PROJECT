"""MJCF world generation for the gamified quadruped environment."""

from __future__ import annotations

from quadruped_locomotion.terrain import (
    ROBOT_CONAFFINITY,
    ROBOT_CONTYPE,
    TerrainSpec,
)


LEG_NAMES = ("fl", "fr", "rl", "rr")
JOINT_NAMES = tuple(f"{leg}_{joint}" for leg in LEG_NAMES for joint in ("hip", "knee", "ankle"))
FOOT_GEOM_NAMES = tuple(f"{leg}_foot" for leg in LEG_NAMES)
ROOT_BODY = "torso"
ROOT_JOINT = "root"

MAX_TORQUE = 45.0
SPAWN_HEIGHT = 0.445

#: How far each foot sits from its own hip along the body axis. Front and rear
#: legs point in opposite directions so the support polygon straddles the
#: centre of mass with equal margin; pointing every foot forward would leave the
#: rear stance only a few centimetres behind the CoM and the robot tips over.
FOOT_OFFSET_X = 0.10

#: Joint rest pose. ``knee``/``ankle`` values were calibrated against the leg
#: geometry so that a standing robot places its feet on the ground at
#: ``SPAWN_HEIGHT`` with the leg links well clear of the floor.
REST_POSE: dict[str, float] = {
    "hip": 0.0,
    "knee": -0.30,
    "ankle": 0.35,
}


def _leg_xml(name: str, x: float, y: float, foot_dx: float) -> str:
    foot_geom = f"{name}_foot"
    foot_site = f"{name}_foot_site"
    return f"""
    <body name="{name}_hip_body" pos="{x:.4f} {y:.4f} 0.0000">
      <joint name="{name}_hip" type="hinge" axis="0 1 0" range="-0.9500 0.9500"
             damping="2.2" armature="0.020" frictionloss="0.05"/>
      <geom name="{name}_upper_leg" type="capsule" fromto="0 0 -0.0150 0 0 -0.1650"
            size="0.0280" mass="0.55" material="robot"/>
      <body name="{name}_knee_body" pos="0 0 -0.1700">
        <joint name="{name}_knee" type="hinge" axis="0 1 0" range="-1.7500 0.7000"
               damping="1.8" armature="0.016" frictionloss="0.04"/>
        <geom name="{name}_lower_leg" type="capsule" fromto="0 0 -0.0050 0 0 -0.1650"
              size="0.0240" mass="0.40" material="robot"/>
        <body name="{name}_ankle_body" pos="0 0 -0.1700">
          <joint name="{name}_ankle" type="hinge" axis="0 1 0" range="-1.6000 1.6000"
                 damping="1.1" armature="0.012" frictionloss="0.02"/>
          <geom name="{name}_foot_link" type="capsule"
                fromto="0.0200 0 -0.0250 {foot_dx:.4f} 0 -0.0550"
                size="0.0180" mass="0.14" material="robot_dark"/>
          <geom name="{foot_geom}" type="box" pos="{foot_dx:.4f} 0 -0.0760"
                size="0.0350 0.0280 0.0120"
                mass="0.16" material="foot" condim="4" friction="0.90 0.005 0.0001"/>
          <site name="{foot_site}" pos="{foot_dx:.4f} 0 -0.0880" size="0.0100" rgba="1 0 0 0"/>
        </body>
      </body>
    </body>
"""


def build_world_xml(terrain: TerrainSpec) -> str:
    """Return the full MJCF for the robot plus a generated track.

    Collision filtering uses a two-group scheme: robot geoms are
    ``contype=1 / conaffinity=2`` and world geoms are ``contype=2 /
    conaffinity=1``. That yields robot-to-ground contacts while making
    robot-to-robot contacts impossible, which avoids the self-collision
    jitter that otherwise dominates torque control.
    """

    actuators = "\n".join(
        f'    <motor name="{joint}_motor" joint="{joint}" gear="1.0"'
        f' ctrlrange="{-MAX_TORQUE:.1f} {MAX_TORQUE:.1f}" ctrllimited="true"'
        f' forcerange="{-MAX_TORQUE * 1.5:.1f} {MAX_TORQUE * 1.5:.1f}"/>'
        for joint in JOINT_NAMES
    )
    legs = "\n".join(
        [
            _leg_xml("fl", 0.20, 0.13, FOOT_OFFSET_X),
            _leg_xml("fr", 0.20, -0.13, FOOT_OFFSET_X),
            _leg_xml("rl", -0.20, 0.13, -FOOT_OFFSET_X),
            _leg_xml("rr", -0.20, -0.13, -FOOT_OFFSET_X),
        ]
    )
    robot_default = (
        f'contype="{ROBOT_CONTYPE}" conaffinity="{ROBOT_CONAFFINITY}"'
        ' condim="4" solref="0.008 1" solimp="0.92 0.98 0.001" friction="0.60 0.005 0.0001"'
    )

    return f"""<mujoco model="gamified_quadruped">
  <compiler angle="radian" inertiafromgeom="auto"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4" cone="elliptic"
          impratio="8" iterations="50" ls_iterations="20"/>
  <size njmax="600" nconmax="200"/>

  <default>
    <geom {robot_default}/>
    <joint limited="true" armature="0.01" damping="1.0"/>
  </default>

  <asset>
    <texture name="grid_tex" type="2d" builtin="checker" rgb1="0.16 0.30 0.24"
             rgb2="0.20 0.36 0.28" width="128" height="128"/>
    <material name="robot" rgba="0.22 0.38 0.88 1" specular="0.4" shininess="0.4"/>
    <material name="robot_dark" rgba="0.07 0.11 0.28 1"/>
    <material name="foot" rgba="0.05 0.05 0.07 1" specular="0.2"/>
    <material name="terrain" rgba="0.30 0.52 0.36 1" specular="0.1"/>
    <material name="uneven" rgba="0.38 0.46 0.30 1" specular="0.1"/>
    <material name="ramp" rgba="0.50 0.40 0.28 1" specular="0.1"/>
    <material name="slippery" rgba="0.55 0.78 0.94 1" specular="0.7" shininess="0.8"/>
    <material name="hazard" rgba="0.90 0.13 0.11 1" specular="0.3"/>
    <material name="platform" rgba="0.86 0.70 0.24 1" specular="0.3"/>
    <material name="void" rgba="0.05 0.06 0.08 1"/>
  </asset>

  <worldbody>
    <light name="sun" pos="-3 -4 7" dir="0.4 0.5 -1" diffuse="0.80 0.80 0.76"
           castshadow="true"/>
    <light name="fill" pos="3 2 4" dir="-0.5 -0.2 -1" diffuse="0.22 0.26 0.34"/>
    <camera name="follow" pos="-3.2 0 1.35" xyaxes="0 -1 0 0.25 0 0.97"/>

    <body name="{ROOT_BODY}" pos="0 0 {SPAWN_HEIGHT:.4f}">
      <freejoint name="{ROOT_JOINT}"/>
      <geom name="torso_shell" type="box" size="0.2750 0.1150 0.0550" mass="7.0" material="robot"/>
      <geom name="torso_deck" type="ellipsoid" pos="0.0250 0 0.0450" size="0.2400 0.1000 0.0320"
            mass="0.6" material="robot_dark"/>
      <site name="imu" pos="0 0 0.0900" size="0.0120" rgba="0 1 0 0"/>
{legs}
    </body>

{terrain.ground_xml}
{terrain.platform_xml}
  </worldbody>

  <actuator>
{actuators}
  </actuator>
</mujoco>
"""