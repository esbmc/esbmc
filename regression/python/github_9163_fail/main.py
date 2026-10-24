import subprocess

r = subprocess.run(["false"], capture_output=True, text=True)
assert r.returncode == 0
