def qualify(modulo: str, name: str) -> str:
    submodulo = f"{modulo}.{name}".lstrip(".")
    return submodulo


assert qualify("", "os") == ".os"
