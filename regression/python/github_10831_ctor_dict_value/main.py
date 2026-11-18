class Box:
    def __init__(self, d: dict[str, float]):
        e = d.get('k')
        assert e is not None
        assert e + 0.5 == 2.0
        assert d['k'] * 2.0 == 3.0


Box({'k': 1.5})
