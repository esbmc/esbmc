import helper

x = [helper.f(), helper.f()]
for i in range(2):
    assert x[i] == "a"
