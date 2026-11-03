import subprocess

r = subprocess.run(["ls"], stdout=subprocess.PIPE)
assert r.stderr is not None
