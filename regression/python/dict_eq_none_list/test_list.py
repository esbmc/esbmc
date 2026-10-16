def test() -> None:
   a ={}
   a.setdefault(1, []).append(1.0)
   assert a == {1: [1.0]}

test()

