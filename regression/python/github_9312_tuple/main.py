import subprocess
def run(x: int) -> int:
    try:
        if x > 0:
            raise subprocess.SubprocessError("boom")
        return 0
    except (OSError, subprocess.SubprocessError):
        return 1
assert run(1) == 1
assert run(0) == 0
