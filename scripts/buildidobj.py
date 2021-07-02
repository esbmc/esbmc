#!/usr/bin/env python3

import sys
import shutil  # For git
import subprocess  # For git
from datetime import datetime  # For date
import getpass  # For username
import socket  # For hostname


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
    def get_last_hash() -> str:
        """Return the hash of the latest commit"""
        git = shutil.which("git")
        if git is None:
            return BuildObj.STR_NOT_GIT

        return subprocess.check_output([git, "rev-parse", "HEAD"]).decode().strip()

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
        git = shutil.which("git")
        if git is None:
            return True

        output = subprocess.check_output(
            [git, "status", "-s"]).decode().splitlines()

        for x in output:
            if '??' not in x:
                return True
        
        return False

    @staticmethod
    def run(output):
        with open(output, 'w') as f:
            f.write(f'{BuildObj.STR_ESBMC_BUILT_FROM} ')
            f.write(f'{BuildObj.get_last_hash()} ')
            f.write(f'{BuildObj.get_datetime()} ')
            f.write(f'{BuildObj.STR_BY} ')
            f.write(f'{BuildObj.get_username()}')
            f.write(f'{BuildObj.STR_AT}')
            f.write(f'{BuildObj.get_hostname()}')
            if BuildObj.is_dirty_tree():
                f.write(f' {BuildObj.STR_DIRTY}')


def main():
    if len(sys.argv) != 2:
        raise ValueError("Program expects <output>")

    output = sys.argv[1]
    BuildObj.run(output)


if __name__ == "__main__":
    main()
