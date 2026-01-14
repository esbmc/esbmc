# Test: Índice negativo fora do limite - deve FALHAR
text = "Python"
c = text[-10]  # IndexError - índice muito negativo
assert c == "P"
