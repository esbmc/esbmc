d = { "key1": [{"name": "item1", "value": "10"}, {"name": "item2", "value": "20"}],
    "key2": [{"name": "item3", "value": "30"}, {"name": "item4", "value": "40"}]
}

k2: list[dict[str, str]] = d["key2"]
for o in k2:
    name = o["name"]
    assert name == "item2" or name == "item4"
