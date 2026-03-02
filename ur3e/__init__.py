"""
UR3e Robot Module

This module provides kinematics and (optionally) PyBullet integration for UR3e robot.
"""

from .ur3e import URRobot

# Lazily import PyBullet integration so headless environments without pybullet can still use kinematics.
try:
    from .ur3e_pybullet import UR3ePyBullet
except ModuleNotFoundError:
    UR3ePyBullet = None

__all__ = ['URRobot', 'UR3ePyBullet']
