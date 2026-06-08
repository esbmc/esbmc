def f() -> int:
    planets = ("Mercury", "Venus", "Earth")
    return planets.index("Earth")


assert f() == 2
