def ioctl(*args, **kwargs):
    raise NotImplementedError("fcntl.ioctl is not supported on Windows; this shim allows imports but not hardware access.")

__all__ = ["ioctl"]
