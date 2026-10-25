import subprocess

r = subprocess.run(["ls"])
assert r.stdout is not None
