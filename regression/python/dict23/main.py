d = {
    "key1": [{
        "name": "item1",
        "value": "10"
    }, {
        "name": "item2",
        "value": "20"
    }],
    "key2": [{
        "name": "item3",
        "value": "30"
    }, {
        "name": "item4",
        "value": "40"
    }]
}

k2 = d["key2"]

# Access first element
name1 = k2[0]["name"]
assert name1 == "item3"

value1 = k2[0]["value"]
assert value1 == "30"

# Access second element
name2 = k2[1]["name"]
assert name2 == "item4"

value2 = k2[1]["value"]
assert value2 == "40"

# Also verify key1 works
k1 = d["key1"]
assert k1[0]["name"] == "item1"
assert k1[0]["value"] == "10"
assert k1[1]["name"] == "item2"
assert k1[1]["value"] == "20"
