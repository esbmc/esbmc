# Test: String imutabilidade - tentativa de modificação deve FALHAR
s = "hello"
s[0] = "H"  # TypeError - strings são imutáveis
assert s == "Hello"
