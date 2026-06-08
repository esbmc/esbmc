from dataclasses import dataclass, field


@dataclass
class Config:
    timeout: int = field(default=30)
    retries: int = field(default=3)


c = Config()
assert c.timeout == 30
assert c.retries == 3

c2 = Config(timeout=60)
assert c2.timeout == 60
assert c2.retries == 3
