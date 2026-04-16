def f() -> int:
    planets = ("Mercury", "Venus", "Earth")
    return planets.index("Venus")


assert f() == 2
