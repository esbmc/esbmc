from dataclasses import dataclass, field


@dataclass
class Config:
    name: str
    retries: int = field(default=3)
    verbose: bool = field(default=False)


c = Config("test")
assert c.retries == 3
assert c.verbose == False
c2 = Config("prod", 5, True)
assert c2.retries == 5
assert c2.verbose == True
