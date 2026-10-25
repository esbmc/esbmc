import subprocess

r = subprocess.run(["ls"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=None)
assert r.stdout is not None
assert r.stderr is None
