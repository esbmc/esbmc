from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    host: str
    port: int = 443


c = Config("example.com")
c2 = Config("dev.io", 8080)
assert c.host == "example.com"
assert c.port == 443
assert c2.port == 8080
