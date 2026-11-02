# #9365: f"{None}" is "None".
x = None
assert f"{x}" == "None"
assert f"a{None}b" == "aNoneb"
