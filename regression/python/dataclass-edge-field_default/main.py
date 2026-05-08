from dataclasses import dataclass, field


@dataclass
class Server:
    host: str
    port: int = 8080
    debug: bool = False


s1 = Server("localhost")
s2 = Server("prod.io", 443, False)
assert s1.port == 8080
assert s1.debug == False
assert s2.port == 443
assert s2.host == "prod.io"
