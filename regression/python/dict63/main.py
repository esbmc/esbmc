d = {"properties": {"x": {"type": "boolean"}}}

for k, v in d["properties"].items():
    if v["type"] == "boolean":
        print(True)
