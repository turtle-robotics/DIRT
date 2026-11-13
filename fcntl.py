"""Windows stand-in for Python's ``fcntl`` module.

``fcntl`` only exists on Unix. Some imports in the project pull it in, so this
stub lets those imports succeed on Windows. Any real call raises.

Warning: because this file sits at the repository root, it also hides the real
``fcntl`` on Linux and the Pi when you run from the root, which breaks
packages such as ``pyserial``. Move or rename it before running on hardware
(see the README's Known issues section).
"""

def ioctl(*args, **kwargs):
    """Always raise ``NotImplementedError``; there is no hardware access on Windows."""
    raise NotImplementedError("fcntl.ioctl is not supported on Windows; this shim allows imports but not hardware access.")

__all__ = ["ioctl"]
