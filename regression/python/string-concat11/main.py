def f(n):
    s = ""
    alphabet = "01"
    while n > 0:
        s = alphabet[n % 2] + s
        n = 0

f(1)
