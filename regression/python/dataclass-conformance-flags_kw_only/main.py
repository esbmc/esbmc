from dataclasses import dataclass


@dataclass(kw_only=True, repr=False, eq=False)
class Config:
    retries: int = 0


cfg = Config(retries=3)
assert cfg.retries == 3
