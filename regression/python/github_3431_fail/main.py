from typing import TypedDict, Dict, Any


class Schema(TypedDict):
    items: Dict[str, Any]


class API:
    def method(self, schema: Schema) -> None:
        pass


api = API()
# Pass a non-dict value to trigger verification failure
x: int = 42
assert x == 0  # This assertion should fail
