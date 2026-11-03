"""The regression bundle: what a shard needs from the producer's build tree.

CTest records absolute paths, so the bundle holds paths relative to the
workspace and is restored into the same absolute workspace. Unit-test
executables stay out: their tests run on the producer.
"""

import json
import os
import re
import shlex
import subprocess

import ci_results

BUNDLE_DIR = "ci-bundle"
_INCLUDE = re.compile(r'^\s*(?:include|subdirs)\("([^"]+)"\)', re.MULTILINE)


class BundleError(Exception):
    pass


def ctest_files(build_dir):
    """Every file CTest reads to list the tree: CTestTestfile.cmake and its includes."""
    found = set()
    pending = [os.path.join(build_dir, "CTestTestfile.cmake")]
    while pending:
        path = os.path.abspath(pending.pop())
        if path in found or not os.path.isfile(path):
            continue
        found.add(path)
        with open(path, encoding="utf-8") as fp:
            for target in _INCLUDE.findall(fp.read()):
                target = os.path.join(os.path.dirname(path), target)
                if os.path.isdir(target):
                    target = os.path.join(target, "CTestTestfile.cmake")
                pending.append(target)
    return found


def _tokens(arg):
    value = arg.partition("=")[2] or arg
    try:
        return shlex.split(value)
    except ValueError:
        return value.split()


def tool_files(plan, build_dir):
    """Files inside the build tree that the shards' test commands name."""
    build_dir = os.path.abspath(build_dir)
    shard_tests = {n for e, names in plan["executors"].items() if e != "producer" for n in names}
    files = set()
    for name in shard_tests:
        for arg in plan["tests"][name]["command"]:
            for token in _tokens(arg):
                path = os.path.abspath(token)
                if path.startswith(build_dir + os.sep) and os.path.isfile(path):
                    files.add(path)
    return files


def _ldd(executable):
    """(library, resolved path or None) for each shared library `executable` needs."""
    completed = subprocess.run(["ldd", executable], capture_output=True, text=True, check=False)
    libraries = []
    for line in completed.stdout.splitlines():
        name, arrow, target = line.strip().partition(" => ")
        if arrow:
            path = target.split(" (")[0].strip()
            libraries.append((name, None if path == "not found" else path))
    return libraries


def library_files(executables, build_dir):
    """Shared libraries the executables load from inside the build tree."""
    build_dir = os.path.abspath(build_dir)
    files = set()
    for executable in executables:
        for _, path in _ldd(executable):
            if path and os.path.abspath(path).startswith(build_dir + os.sep):
                files.update({os.path.abspath(path), os.path.realpath(path)})
    return files


def bundle_files(workspace, build_dir, plan):
    tools = tool_files(plan, build_dir)
    files = ctest_files(build_dir) | tools | library_files(tools, build_dir)
    relative = sorted(os.path.relpath(f, workspace) for f in files)
    outside = [f for f in relative if f.startswith("..")]
    if outside:
        raise BundleError(f"bundle files outside the workspace: {outside[:5]}")
    return relative


def create(workspace, build_dir, plan, metadata, out_path):
    """Write the bundle and return its SHA-256."""
    os.makedirs(os.path.join(workspace, BUNDLE_DIR), exist_ok=True)
    meta = dict(metadata,
                workspace=os.path.abspath(workspace),
                plan_digest=plan["digest"],
                executables=sorted(
                    os.path.relpath(f, workspace) for f in tool_files(plan, build_dir)))
    with open(os.path.join(workspace, BUNDLE_DIR, "metadata.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=1, sort_keys=True)
    members = bundle_files(workspace, build_dir, plan) + [BUNDLE_DIR]
    subprocess.run(["tar", "-I", "zstd -3 -T0", "-cf", out_path, "-C", workspace, *members],
                   check=True)
    return ci_results.file_digest(out_path)


def _check_members(tar_path):
    names = subprocess.run(["tar", "-I", "zstd", "-tf", tar_path],
                           check=True,
                           capture_output=True,
                           text=True).stdout.split()
    unsafe = [n for n in names if os.path.isabs(n) or ".." in n.split("/")]
    if unsafe:
        raise BundleError(f"bundle member paths escape the workspace: {unsafe[:5]}")


def restore(tar_path, expected_sha256, workspace):
    """Check and extract the bundle; return its metadata."""
    digest = ci_results.file_digest(tar_path)
    if digest != expected_sha256:
        raise BundleError(f"bundle digest {digest} is not the producer's {expected_sha256}")
    _check_members(tar_path)
    subprocess.run(["tar", "-I", "zstd", "-xf", tar_path, "-C", workspace], check=True)
    with open(os.path.join(workspace, BUNDLE_DIR, "metadata.json"), encoding="utf-8") as fp:
        meta = json.load(fp)
    if meta["workspace"] != os.path.abspath(workspace):
        raise BundleError(f"bundle was built in {meta['workspace']}, CTest needs that path; "
                          f"this runner restores into {os.path.abspath(workspace)}")
    for executable in meta["executables"]:
        path = os.path.join(workspace, executable)
        if not os.access(path, os.X_OK):
            raise BundleError(f"{executable} is missing or not executable after restore")
        missing = [name for name, found in _ldd(path) if found is None]
        if missing:
            raise BundleError(f"{executable} cannot load {', '.join(missing)} on this runner")
    return meta
