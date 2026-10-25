import subprocess

r = subprocess.run(["false"], check=True)
assert r.returncode == 0
