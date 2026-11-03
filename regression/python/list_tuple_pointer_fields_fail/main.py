def entries(top: str) -> list[tuple[str, list[str], list[str]]]:
    result: list[tuple[str, list[str], list[str]]] = []
    dirs: list[str] = ["a"]
    files: list[str] = []
    result.append((top, dirs, files))
    return result


for path, ds, fs in entries("x"):
    assert path == "x"
    assert len(ds) == 1
    assert len(fs) == 1
