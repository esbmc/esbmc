from dataclasses import InitVar, dataclass


@dataclass
class Token:
    text: str
    suffix: InitVar[str]

    def __post_init__(self, suffix) -> None:
        self.text = self.text + "-" + suffix


t = Token("id", "ok")

assert t.text == "id-ok"
