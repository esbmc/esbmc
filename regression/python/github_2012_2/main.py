from typing import List


class Service:
    def search(self) -> List[int]:
        return [1, 2, 3]


svc: Service


def connect() -> bool:
    global svc
    svc = Service()
    return True


def collect() -> List[int]:
    global svc
    values = svc.search()
    first = values[0]
    assert first == 1
    return values


assert connect()
result = collect()
assert result[2] == 3
