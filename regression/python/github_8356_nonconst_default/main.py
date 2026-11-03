class V:
    def push(self, x: int, xs: list[int] = []) -> int:
        xs.append(x)
        return len(xs)

    add = push

v = V()
assert v.push(1) == 1
assert v.add(2) == 2
