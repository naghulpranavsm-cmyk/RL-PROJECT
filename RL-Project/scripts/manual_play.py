"""Interactive GLFW play mode for the gamified MuJoCo quadruped."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import traceback

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import glfw
import mujoco

from quadruped_locomotion import QuadrupedGameEnv


CONTROLS_HELP = (
    "WASD / Arrows  move\n"
    "Space          jump\n"
    "Shift          sprint\n"
    "R              reset pose\n"
    "N              new track\n"
    "P              pause\n"
    "Esc            quit"
)


def _health_bar(health: float, width: int = 22) -> str:
    filled = int(round(max(0.0, min(100.0, health)) / 100.0 * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _command_from_keys(pressed: set[int]) -> dict[str, float | bool]:
    """Map the held keys to a command.

    Only the analogue axes are read from held state. Reset and jump are edge
    events handled by the caller, otherwise a held key would repeat them every
    frame and a quick tap could fall between two frames and be lost.
    """

    forward = 0.0
    turn = 0.0
    if glfw.KEY_W in pressed or glfw.KEY_UP in pressed:
        forward += 1.0
    if glfw.KEY_S in pressed or glfw.KEY_DOWN in pressed:
        forward -= 1.0
    if glfw.KEY_A in pressed or glfw.KEY_LEFT in pressed:
        turn += 1.0
    if glfw.KEY_D in pressed or glfw.KEY_RIGHT in pressed:
        turn -= 1.0
    return {
        "forward": forward,
        "turn": turn,
        "jump": False,
        "sprint": glfw.KEY_LEFT_SHIFT in pressed or glfw.KEY_RIGHT_SHIFT in pressed,
        "reset": False,
    }


def _hud_text(info: dict, reward: float, paused: bool) -> tuple[str, str]:
    header = "PAUSED" if paused else f"LEVEL {int(info['level'])}"
    left = (
        f"{header}\n"
        f"Score:  {info['score']:9.1f}\n"
        f"Health: {_health_bar(float(info['health']))} {info['health']:5.1f}\n"
        f"Combo:  x{info['combo_multiplier']:.2f}  ({int(info['combo'])})\n"
        f"Reward: {reward:+.3f}\n"
        f"Dist:   {info['x_position']:6.1f} / {info['track_length']:.0f} m\n"
        f"Falls:  {int(info['falls'])}   Hits: {int(info['collisions'])}"
    )
    return left, CONTROLS_HELP


class Viewer:
    """Owns the GLFW window and re-creates MuJoCo scene state after a track rebuild."""

    def __init__(self, env: QuadrupedGameEnv, width: int, height: int) -> None:
        if not glfw.init():
            raise SystemExit("GLFW failed to initialize; the viewer cannot run on this machine.")
        self.window: int | None = glfw.create_window(
            width, height, "MuJoCo Quadruped Locomotion Game", None, None
        )
        if self.window is None:
            glfw.terminate()
            raise SystemExit("Could not create a GLFW window.")
        assert self.window is not None
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.camera = mujoco.MjvCamera()
        self.options = mujoco.MjvOption()
        self.scene: mujoco.MjvScene | None = None
        self.context: mujoco.MjrContext | None = None
        self._model_version = -1

    def sync(self, env: QuadrupedGameEnv) -> None:
        if self.scene is not None and self._model_version == env.model_version:
            return
        self._release()
        self.scene = mujoco.MjvScene(env.model, maxgeom=20_000)
        self.context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self._model_version = env.model_version

    def draw(self, env: QuadrupedGameEnv, info: dict, reward: float, paused: bool) -> None:
        assert self.window is not None
        assert self.scene is not None and self.context is not None
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, fb_w, fb_h)
        env.follow_camera(self.camera)
        mujoco.mjv_updateScene(
            env.model,
            env.data,
            self.options,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        left, right = _hud_text(info, reward, paused)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            left,
            right,
            self.context,
        )
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def _release(self) -> None:
        """Release the render resources for the current model.

        ``MjvScene`` owns no GL resources; its arrays are owned by the Python
        wrapper and are reclaimed by refcounting, so it must not be freed by
        hand (``MjvScene`` has no ``free`` method). ``MjrContext`` does own GL
        objects -- a framebuffer, textures, and the built-in fonts -- and must be
        freed exactly once, while the window's GL context is still current,
        i.e. before ``glfw.destroy_window``.
        """

        context, self.context = self.context, None
        self.scene = None
        if context is not None:
            context.free()

    def close(self) -> None:
        self._release()
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--track-length", type=float, default=80.0)
    parser.add_argument("--reward-profile", choices=("spec", "balanced"), default="spec")
    args = parser.parse_args()

    env = QuadrupedGameEnv(
        seed=args.seed,
        start_level=args.level,
        track_length=args.track_length,
        reward_profile=args.reward_profile,
    )
    viewer = Viewer(env, args.width, args.height)

    pressed: set[int] = set()
    toggles: set[int] = set()
    paused = False

    def key_callback(handle: int, key: int, scancode: int, action: int, mods: int) -> None:
        del scancode, mods
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(handle, True)
        elif key == glfw.KEY_P and action == glfw.PRESS:
            toggles.add("pause")
        elif key == glfw.KEY_N and action == glfw.PRESS:
            toggles.add("new_track")
        elif action == glfw.PRESS:
            pressed.add(key)
            if key in (glfw.KEY_R, glfw.KEY_SPACE):
                toggles.add(key)
        elif action == glfw.RELEASE:
            pressed.discard(key)

    glfw.set_key_callback(viewer.window, key_callback)

    reward = 0.0
    info = env.info()
    frame_dt = 1.0 / env.metadata["render_fps"]

    try:
        while not glfw.window_should_close(viewer.window):
            loop_start = time.perf_counter()

            if "pause" in toggles:
                paused = not paused
                toggles.discard("pause")

            if "new_track" in toggles:
                toggles.discard("new_track")
                level = env.game.level + 1
                env.reset(options={"new_track": True, "level": level})
                reward = 0.0
                info = env.info()

            if not paused:
                command = _command_from_keys(pressed)
                if glfw.KEY_R in toggles:
                    command["reset"] = True
                    toggles.discard(glfw.KEY_R)
                if glfw.KEY_SPACE in toggles:
                    command["jump"] = True
                    toggles.discard(glfw.KEY_SPACE)

                _, reward, terminated, truncated, info = env.step(command)
                if terminated or truncated:
                    env.reset()
                    reward = 0.0
                    info = env.info()

            viewer.sync(env)
            viewer.draw(env, info, reward, paused)

            elapsed = time.perf_counter() - loop_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    finally:
        # Report, but never raise, cleanup failures while another exception is
        # already propagating: a teardown bug must not hide the real error.
        propagating = sys.exc_info()[0] is not None
        for cleanup in (env.close, viewer.close):
            try:
                cleanup()
            except Exception:
                if not propagating:
                    raise
                print("warning: cleanup failed", file=sys.stderr)
                traceback.print_exc()


if __name__ == "__main__":
    main()