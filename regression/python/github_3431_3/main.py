# Test TypedDict used as return type
from typing import TypedDict


class Config(TypedDict):
    debug: bool
    port: int


def get_config() -> Config:
    return {"debug": True, "port": 8080}


cfg = get_config()
assert cfg is not None
