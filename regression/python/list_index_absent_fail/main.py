# list.index(x) raises ValueError when x is absent (modelled as a failing
# assertion inside the operational model).
a = [1, 2, 3]
i = a.index(9)
