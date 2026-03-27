class API:

    def method(self, mode: str, schema: dict) -> dict:
        return {"result": True}


api = API()
result = api.method(mode={"invalid": "dict"}, schema={"properties": {"field": {"type": "boolean"}}})
