# Issue #9312: an except clause naming module.Class.
import subprocess

try:
    r = subprocess.run(["ls"], check=True)
    rc = r.returncode
except subprocess.SubprocessError:
    rc = -1
assert rc == 0 or rc == -1
