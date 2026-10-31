import subprocess
try:
    raise subprocess.CalledProcessError(3, "ls")
except subprocess.CalledProcessError as e:
    assert e.returncode == 3
