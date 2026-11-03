people = [{"name": "bob", "age": 30}, {"name": "amy", "age": 25}]
s = sorted(people, key=lambda p: p["name"])
assert s[0]["age"] == 25
assert s[1]["name"] == "bob"
