a = 60
b = 5

sum_f = float(a + b)
assert sum_f  == 65.0

sun_i = int(sum_f)
assert sun_i == 65

#FIXME: chr doesn't work with variables.
#We should model calls to builtin_type functions (e.g: int(x)) as typecasts
#sun_char = chr(sun_i)
#assert sun_char == 'A'

sun_char = chr(65)
assert sun_char == 'A'
