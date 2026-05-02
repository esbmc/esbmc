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


def mkdir(path: str) -> None:
    dir_not_exists: bool = nondet_bool()
    if not dir_not_exists:
        raise FileExistsError("Directory already exists")


def rmdir(path: str) -> None:
    is_empty: bool = nondet_bool()
    if not is_empty:
        raise OSError("Directory not empty")