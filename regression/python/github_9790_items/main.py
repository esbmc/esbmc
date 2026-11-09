c = {1: 2, 2: "x"}
for k, v in c.items():
    assert k != 2 or v == "x"
    assert k != 1 or v == 2
