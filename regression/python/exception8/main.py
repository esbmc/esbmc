def open_file(filename: str) -> None:
    if filename == "":
        raise FileNotFoundError("Empty filename")


try:
    open_file("")
except OSError as e:
    print("Caught by OSError:", e)
