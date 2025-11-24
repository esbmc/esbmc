test1 = str('test')
assert test1 == 'test'        # same content
assert test1 != 'other'       # different content
assert test1 != 'testing'     # same prefix, longer string
assert 'test' == test1        # reversed comparison
assert not (test1 != 'test')  # ensure != is false for equal strings
