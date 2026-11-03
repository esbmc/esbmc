import subprocess
from subprocess import CalledProcessError

try:
    r = subprocess.run(["ls"], capture_output=True, text=True, check=True)
    assert r.returncode == 0
    r.check_returncode()
except CalledProcessError as e:
    assert e.returncode != 0
