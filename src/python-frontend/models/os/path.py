# pylint: disable=undefined-variable,unused-argument
# Operational-model stubs: argument names are part of the API contract
# matched by ESBMC's Python converter, even when the body does not
# reference them. `nondet_bool` and `nondet_str` are ESBMC intrinsics.

# Stubs for os.path module


def exists(path: str) -> bool:
    return False


def basename(path: str) -> str:
    bname: str = ""
    return bname


def join(a: str, b: str) -> str:
    if b.startswith("/"):
        return b
    if a == "" or a.endswith("/"):
        return a + b
    return a + "/" + b


def islink(path: str) -> bool:
    return nondet_bool()


# realpath, relpath and commonpath depend on the file system or on
# normalisation the model does not perform, so their results are unknown.
def realpath(path: str) -> str:
    return nondet_str()


def relpath(path: str, start: str = ".") -> str:
    if path == "":
        raise ValueError("no path specified")
    return nondet_str()


def commonpath(paths: list[str]) -> str:
    if len(paths) == 0:
        raise ValueError("commonpath() arg is an empty sequence")
    absolute: bool = paths[0].startswith("/")
    i: int = 1
    while i < len(paths):
        if paths[i].startswith("/") != absolute:
            raise ValueError("Can't mix absolute and relative paths")
        i += 1
    return nondet_str()
