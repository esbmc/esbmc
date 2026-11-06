try:
    with open("missing.txt") as f:
        assert len(f) >= 0
except OSError:
    pass
