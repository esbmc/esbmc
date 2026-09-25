#!/usr/bin/env python3

import sys, os
import re  # For comparing build IDs
import shutil  # For git
import subprocess  # For git
from datetime import datetime  # For date
import getpass  # For username
import socket  # For hostname
from typing import Union


class BuildObj:
    """
        Build Obj will generate an string representing the current release

        The output file should be in the format:

        "ESBMC built from <git-hash> <date> by <username>@<hostname> (dirty-tree)?" 
    """
    STR_ESBMC_BUILT_FROM = "ESBMC built from "
    STR_BY = "by"
    STR_AT = "@"
    STR_DIRTY = "(dirty tree)"
    STR_NOT_GIT = "no-hash"

    @staticmethod
    def try_git_command(git_command_args: list[str]) -> Union[str, bool]:
        git = shutil.which("git")
        if git is None:
            return False

        try:
            cmd = [git]
            cmd.extend(git_command_args)
            return subprocess.check_output(cmd, cwd=os.path.dirname(__file__)).decode()
        except subprocess.CalledProcessError as e:
            # git rev-parse returns 128 if not in a git repository
            if e.returncode == 128:
                return False
            else:
                raise e

    @staticmethod
    def get_last_hash() -> str:
        """Return the hash of the latest commit"""
        result = BuildObj.try_git_command(["rev-parse", "HEAD"])
        if isinstance(result, str):
            return result.strip()
        return BuildObj.STR_NOT_GIT

    @staticmethod
    def get_datetime() -> str:
        """Try to simulate the `date` command"""
        output = datetime.now()
        return str(output)

    @staticmethod
    def get_username() -> str:
        return str(getpass.getuser())

    @staticmethod
    def get_hostname() -> str:
        return str(socket.gethostname())

    @staticmethod
    def is_dirty_tree() -> bool:
        result = BuildObj.try_git_command(["status", "-s"])
        if isinstance(result, str):
            for x in result.splitlines():
                if "??" not in x:
                    return True

            return False
        return True

    @staticmethod
    def format_id() -> str:
        line = (f'{BuildObj.STR_ESBMC_BUILT_FROM} {BuildObj.get_last_hash()} '
                f'{BuildObj.get_datetime()} {BuildObj.STR_BY} '
                f'{BuildObj.get_username()}{BuildObj.STR_AT}'
                f'{BuildObj.get_hostname()}')
        if BuildObj.is_dirty_tree():
            line += f' {BuildObj.STR_DIRTY}'
        return line

    @staticmethod
    def without_datetime(line: str) -> str:
        return re.sub(r'\d{4}-\d{2}-\d{2} [\d:.]+ ', '', line)

    @staticmethod
    def describes_same_build(output, line) -> bool:
        """The build system re-runs this on every build so the ID cannot go
        stale. Rewriting unconditionally would then relink the whole binary
        every time, so report when the only difference is the timestamp."""
        # errors='replace' rather than a narrower except: a file written by an
        # older revision carries the locale encoding, and a decode failure here
        # would abort the build. Mangled text simply reads as a different ID.
        try:
            with open(output, encoding='utf-8', errors='replace') as f:
                old = f.read()
        except OSError:
            return False
        return BuildObj.without_datetime(old) == BuildObj.without_datetime(line)

    @staticmethod
    def run(output):
        line = BuildObj.format_id()
        if BuildObj.describes_same_build(output, line):
            return
        with open(output, 'w', encoding='utf-8') as f:
            f.write(line)


def main():
    if len(sys.argv) != 2:
        raise ValueError("Program expects <output>")

    output = sys.argv[1]
    BuildObj.run(output)


if __name__ == "__main__":
    main()
