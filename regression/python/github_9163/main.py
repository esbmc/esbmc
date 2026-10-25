import subprocess


def g(c: list) -> int:
    r = subprocess.run(c, capture_output=True, text=True)
    return r.returncode


rc = g(["ls"])
assert isinstance(rc, int)
