# GitHub #9086: float() of None
from typing import Optional

status: Optional[str] = None
code = float(status)
