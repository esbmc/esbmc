# Test: Caracteres especiais e dígitos - SUCESSO
digit = "5"
assert digit.isdigit()
assert not digit.isalpha()
space = " "
assert space.isspace()
newline = "\n"
assert len(newline) == 1
tab = "\t"
assert len(tab) == 1
# Comparação com dígitos
assert "0" < "9"
assert "5" > "2"
