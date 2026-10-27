# pylint: disable=undefined-variable,unused-argument
# `nondet_bool` is an ESBMC intrinsic matched by name by the Python
# converter; it has no Python binding. Argument names on the abstract
# stubs below are part of the API contract matched by ESBMC's Python
# converter, even when the body does not reference them.

# Stubs for os module - operating system interfaces


def popen(path: str) -> None:
    pass


def listdir(path: str) -> list[str]:
    directories: list[str] = ["foo", "bar"]
    return directories


def makedirs(path: str, exist_ok: bool = False) -> None:
    if not exist_ok:
        dir_exists: bool = nondet_bool()
        if dir_exists:
            raise FileExistsError("File exists")


def remove(path: str) -> None:
    file_exists: bool = nondet_bool()
    if not file_exists:
        raise FileNotFoundError("No such file or directory")


def unlink(path: str) -> None:
    remove(path)


def mkdir(path: str) -> None:
    dir_not_exists: bool = nondet_bool()
    if not dir_not_exists:
        raise FileExistsError("Directory already exists")


def rmdir(path: str) -> None:
    is_empty: bool = nondet_bool()
    if not is_empty:
        raise OSError("Directory not empty")


def walk(top: str) -> list[tuple[str, list[str], list[str]]]:
    # Yields nothing (top missing or not a directory) or top itself as an
    # empty directory; nondet entry names make the str copies unbounded.
    entries: list[tuple[str, list[str], list[str]]] = []
    if nondet_bool():
        dirnames: list[str] = []
        filenames: list[str] = []
        entries.append((top, dirnames, filenames))
    return entries
