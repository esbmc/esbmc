# Test f-string with member access outside of __str__
class Config:
    def __init__(self, host: str, port: str):
        self.host = host
        self.port = port

c = Config("localhost", "8080")
result: str = f"{c.host}:{c.port}"
assert result == "localhost:8080"
