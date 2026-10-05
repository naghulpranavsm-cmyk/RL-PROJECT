"""Procedural terrain, hazard, and moving-platform generation.

Every generated feature is mirrored in a compact analytic description so the
environment can query the ground height and gap coverage at an arbitrary ``x``
without ray casting. That is used for observations, respawn safety checks, and
difficulty-aware terrain limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.sax.saxutils import escape

import numpy as np


TILE_LENGTH = 1.25
LANE_HALF_WIDTH = 1.15
START_PAD = 6.0
GROUND_THICKNESS = 0.045
GROUND_TOP = 0.005

GRIP_FRICTION = (1.20, 0.005, 0.0001)
SLIP_FRICTION = (0.03, 0.0008, 0.00002)

ROBOT_CONTYPE = 1
ROBOT_CONAFFINITY = 2
WORLD_CONTYPE = 2
WORLD_CONAFFINITY = 1


@dataclass(frozen=True)
class Hazard:
    """A static obstacle placed on the track."""

    name: str
    kind: str
    x: float
    y: float
    clearance_x: float


@dataclass(frozen=True)
class MovingPlatform:
    """A platform that slides laterally, driven kinematically by the env."""

    name: str
    joint_name: str
    base_y: float
    amplitude: float
    speed: float
    phase: float
    x: float


@dataclass
class TerrainSpec:
    """MJCF fragments plus the analytic description of the generated track."""

    ground_xml: str = ""
    platform_xml: str = ""
    hazards: list[Hazard] = field(default_factory=list)
    moving_platforms: list[MovingPlatform] = field(default_factory=list)

    x_min: float = -START_PAD
    x_max: float = 0.0
    segments_x0: np.ndarray = field(default_factory=lambda: np.zeros(0))
    segments_x1: np.ndarray = field(default_factory=lambda: np.zeros(0))
    segments_top: np.ndarray = field(default_factory=lambda: np.zeros(0))
    segments_slope: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gap_x0: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gap_x1: np.ndarray = field(default_factory=lambda: np.zeros(0))
    max_gap_width: float = 0.0

    @property
    def length(self) -> float:
        return float(self.x_max - self.x_min)

    def is_gap(self, x: float) -> bool:
        if self.gap_x0.size == 0:
            return False
        idx = int(np.searchsorted(self.gap_x1, x, side="right"))
        if idx >= self.gap_x0.size:
            return False
        return bool(self.gap_x0[idx] <= x <= self.gap_x1[idx])

    def ground_height(self, x: float) -> float | None:
        """Top surface height at ``x``, or ``None`` when ``x`` is over a gap."""
        if self.segments_x0.size == 0:
            return None
        if x < self.segments_x0[0] or x > self.segments_x1[-1]:
            return None
        idx = int(np.searchsorted(self.segments_x1, x, side="right"))
        idx = min(max(idx, 0), self.segments_x0.size - 1)
        # A gap leaves a hole between two segments and searchsorted lands on the
        # segment after it, so confirm ``x`` is really covered by this one.
        if x < self.segments_x0[idx]:
            return None
        centre = 0.5 * (self.segments_x0[idx] + self.segments_x1[idx])
        top = self.segments_top[idx] + self.segments_slope[idx] * (x - centre)
        return float(top)

    def has_ground(self, x: float) -> bool:
        return self.ground_height(x) is not None


def _attr_geom(
    *,
    name: str,
    geom_type: str,
    pos: tuple[float, float, float],
    size: tuple[float, ...],
    material: str,
    friction: tuple[float, float, float],
    condim: int,
    euler: tuple[float, float, float] | None = None,
) -> str:
    euler_attr = "" if euler is None else f' euler="{euler[0]:.4f} {euler[1]:.4f} {euler[2]:.4f}"'
    size_str = " ".join(f"{value:.4f}" for value in size)
    return (
        f'<geom name="{escape(name)}" type="{geom_type}"'
        f' pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" size="{size_str}"'
        f' material="{material}"'
        f' friction="{friction[0]:.4f} {friction[1]:.4f} {friction[2]:.4f}"'
        f' condim="{condim}" contype="{WORLD_CONTYPE}" conaffinity="{WORLD_CONAFFINITY}"'
        f' solref="0.008 1" solimp="0.9 0.97 0.001"{euler_attr}/>\n'
    )


def _level_scaling(level: int) -> float:
    return 1.0 + 0.18 * max(0, level - 1)


def generate_terrain(
    seed: int | None,
    level: int,
    length: float = 80.0,
    *,
    max_gap_width: float | None = None,
) -> TerrainSpec:
    """Build one track for ``level``.

    Difficulty grows both with ``level`` and with distance along the track. The
    first few metres are deliberately simple so resets and learning stay stable.
    Gap widths are bounded so every gap stays clearable by walking or jumping.
    """

    rng = np.random.default_rng(seed)
    level = max(1, int(level))
    scale = _level_scaling(level)
    if max_gap_width is None:
        max_gap_width = min(0.32 + 0.05 * (level - 1), 0.62)

    ground_xml: list[str] = []
    platform_xml: list[str] = []
    hazards: list[Hazard] = []
    platforms: list[MovingPlatform] = []

    seg_x0: list[float] = []
    seg_x1: list[float] = []
    seg_top: list[float] = []
    seg_slope: list[float] = []
    gap_x0: list[float] = []
    gap_x1: list[float] = []

    index = 0
    x = -START_PAD
    x_max = length
    previous_was_gap = False

    ground_xml.append(
        _attr_geom(
            name="terrain_catch_floor",
            geom_type="box",
            pos=(0.5 * (-START_PAD + length), 0.0, -3.0),
            size=(0.5 * (length + START_PAD) + 6.0, 9.0, 0.20),
            material="void",
            friction=(0.9, 0.005, 0.0001),
            condim=3,
        )
    )

    while x < x_max:
        centre = x + 0.5 * TILE_LENGTH
        progress = max(0.0, (centre - START_PAD) / max(1.0, length - START_PAD))
        tier = 1 + level + int(6 * progress)
        d = float(scale)

        is_start_zone = centre < 3.0
        gap_chance = min(0.04 + 0.012 * tier, 0.16) * d
        ramp_chance = min(0.10 + 0.014 * tier, 0.26) * d
        slippery_chance = min(0.06 + 0.010 * tier, 0.20) * d
        uneven_chance = min(0.16 + 0.014 * tier, 0.34) * d

        roll = rng.random()
        want_gap = (
            not is_start_zone
            and not previous_was_gap
            and centre > 6.0
            and roll < gap_chance
        )

        if want_gap:
            width = float(rng.uniform(0.26, max_gap_width))
            gap_x0.append(x)
            gap_x1.append(x + width)
            previous_was_gap = True
            x += width
            index += 1
            continue

        height = 0.0
        slope = 0.0
        material = "terrain"
        friction = GRIP_FRICTION
        condim = 4
        euler: tuple[float, float, float] | None = None

        if not is_start_zone and roll < gap_chance + ramp_chance:
            angle = float(rng.uniform(-0.20, 0.24))
            slope = float(np.tan(angle))
            height = float(rng.uniform(0.01, 0.05))
            material = "ramp"
            euler = (0.0, angle, 0.0)
        elif not is_start_zone and roll < gap_chance + ramp_chance + slippery_chance:
            material = "slippery"
            friction = SLIP_FRICTION
            condim = 3
        elif not is_start_zone and roll < gap_chance + ramp_chance + slippery_chance + uneven_chance:
            height = float(rng.uniform(-0.025, 0.050))
            material = "uneven"

        top_at_centre = GROUND_TOP + height
        ground_xml.append(
            _attr_geom(
                name=f"terrain_tile_{index}",
                geom_type="box",
                pos=(centre, 0.0, top_at_centre - GROUND_THICKNESS),
                size=(0.5 * TILE_LENGTH, LANE_HALF_WIDTH, GROUND_THICKNESS),
                material=material,
                friction=friction,
                condim=condim,
                euler=euler,
            )
        )
        seg_x0.append(x)
        seg_x1.append(x + TILE_LENGTH)
        seg_top.append(top_at_centre)
        seg_slope.append(slope)

        hazard_chance = min(0.09 + 0.012 * tier, 0.26) * d
        if not is_start_zone and rng.random() < hazard_chance:
            if rng.random() < 0.55:
                name = f"hazard_spike_{index}"
                y = float(rng.uniform(-0.60, 0.60))
                base_z = top_at_centre
                ground_xml.append(
                    _attr_geom(
                        name=name,
                        geom_type="box",
                        pos=(centre, y, base_z + 0.14),
                        size=(0.075, 0.075, 0.14),
                        material="hazard",
                        friction=(1.0, 0.004, 0.0001),
                        condim=3,
                        euler=(0.0, 0.0, float(rng.uniform(0.0, np.pi))),
                    )
                )
                hazards.append(
                    Hazard(name=name, kind="spike", x=centre, y=y, clearance_x=centre + 0.55)
                )
            else:
                name = f"hazard_wall_{index}"
                # Keep the inner face at |y| >= 0.52 so the wall is a barrier to
                # dodge, not a plug across the running lane.
                y = float(rng.choice([-0.82, 0.82]))
                height_m = float(rng.uniform(0.22, 0.30))
                ground_xml.append(
                    _attr_geom(
                        name=name,
                        geom_type="box",
                        pos=(centre, y, top_at_centre + height_m),
                        size=(0.08, 0.30, height_m),
                        material="hazard",
                        friction=(1.0, 0.005, 0.0001),
                        condim=3,
                    )
                )
                hazards.append(
                    Hazard(name=name, kind="wall", x=centre, y=y, clearance_x=centre + 0.60)
                )

        platform_chance = min(0.02 + 0.005 * tier, 0.10) * d
        if not is_start_zone and rng.random() < platform_chance:
            name = f"moving_platform_{index}"
            joint_name = f"{name}_slide"
            base_y = float(rng.uniform(-0.22, 0.22))
            amplitude = float(rng.uniform(0.20, 0.46))
            speed = float(rng.uniform(0.45, 0.95)) * scale
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            platform_xml.append(
                f'<body name="{name}" pos="{centre:.4f} {base_y:.4f} {top_at_centre + 0.055:.4f}">\n'
                f'  <joint name="{joint_name}" type="slide" axis="0 1 0" damping="0.0" armature="0.01" limited="false"/>\n'
                f'  <geom name="{name}_geom" type="box" size="0.4200 0.3000 0.0550" material="platform"'
                f' friction="1.3000 0.0050 0.0001" condim="4"'
                f' contype="{WORLD_CONTYPE}" conaffinity="{WORLD_CONAFFINITY}"/>\n'
                f'</body>\n'
            )
            platforms.append(
                MovingPlatform(
                    name=name,
                    joint_name=joint_name,
                    base_y=base_y,
                    amplitude=amplitude,
                    speed=speed,
                    phase=phase,
                    x=centre,
                )
            )

        previous_was_gap = False
        x += TILE_LENGTH
        index += 1

    spec = TerrainSpec(
        ground_xml="".join(ground_xml),
        platform_xml="".join(platform_xml),
        hazards=hazards,
        moving_platforms=platforms,
        x_min=-START_PAD,
        x_max=x_max,
        segments_x0=np.asarray(seg_x0, dtype=np.float64),
        segments_x1=np.asarray(seg_x1, dtype=np.float64),
        segments_top=np.asarray(seg_top, dtype=np.float64),
        segments_slope=np.asarray(seg_slope, dtype=np.float64),
        gap_x0=np.asarray(gap_x0, dtype=np.float64),
        gap_x1=np.asarray(gap_x1, dtype=np.float64),
        max_gap_width=float(max_gap_width),
    )
    return spec