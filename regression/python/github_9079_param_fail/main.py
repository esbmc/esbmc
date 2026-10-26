# GitHub #9079: through a parameter
from typing import Optional


def match(stderr: Optional[str]) -> bool:
    return 'ab' in stderr


match(None)
