from typing import Optional

# GitHub #9086: int() of None raises TypeError.

status: Optional[str] = None
code = int(status)
