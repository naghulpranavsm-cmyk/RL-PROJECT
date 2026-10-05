"""Gamified MuJoCo quadruped locomotion environment."""

__all__ = ["QuadrupedGameEnv"]


def __getattr__(name: str):
    if name == "QuadrupedGameEnv":
        from quadruped_locomotion.env import QuadrupedGameEnv

        return QuadrupedGameEnv
    raise AttributeError(name)
