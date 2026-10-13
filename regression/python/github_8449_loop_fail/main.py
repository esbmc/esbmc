i = 0
while i < 2:
    if i == 0:
        assert k() == 3

    def k() -> int:
        return 3

    i += 1
