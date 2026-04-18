#!/bin/bash
# Test Homebrew formula update logic

set -e

pip3 install --quiet --break-system-packages requests packaging 2>/dev/null || true

python3 << 'EOF'
import re, requests
from packaging import version as pkg_version

try:
    with open('Formula/esbmc.rb') as f:
        formula = f.read()
except FileNotFoundError:
    print("Error: Run from repository root")
    exit(1)

current = re.search(r'version\s+"([^"]+)"', formula).group(1)
latest = requests.get("https://api.github.com/repos/esbmc/esbmc/releases/latest").json()['tag_name'].lstrip('v')

get_base = lambda v: re.search(r'(\d+\.\d+)', v).group(1)
current_base, latest_base = get_base(current), get_base(latest)

print(f"Current: {current} (base: {current_base})")
print(f"Latest:  {latest} (base: {latest_base})")

if pkg_version.parse(latest_base) > pkg_version.parse(current_base):
    print(f"\n✅ Update available: {current_base} -> {latest_base}")
else:
    print(f"\n✅ Up to date")

EOF
