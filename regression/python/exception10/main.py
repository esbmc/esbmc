class CustomError(Exception):
    message: str = ""

    def __init__(self, message: str):
        self.message = message


class SpecificError(CustomError):
    message: str = ""

    def __init__(self, message: str):
        self.message = message


def process() -> None:
    raise SpecificError("Specific issue")


try:
    process()
except CustomError as e:
    print("Caught by CustomError:", e)
