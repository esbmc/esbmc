# Issue #10130: `t or t[0] == "a"` evaluates t[0] when t is empty.
t = ""
r = t or t[0] == "a"
