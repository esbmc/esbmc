assert "a" + "b" == "ab"
assert "" + "" == ""
assert "hello" + "" == "hello"
assert "" + "world" == "world"

x = "foo"
y = "bar"
assert x + y == "foobar"
assert x + "-" + y == "foo-bar"

assert "hello" + " " + "world" == "hello world"
assert "x" + "," + "y" == "x,y"

assert "α" + "β" == "αβ"
assert "😀" + "🐍" == "😀🐍"
