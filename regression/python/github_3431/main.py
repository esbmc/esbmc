from typing import TypedDict, Dict, Any


class Schema(TypedDict):
    items: Dict[str, Any]


class API:
    def method(self, schema: Schema) -> None:
        pass


api = API()
api.method({"items": {}})
