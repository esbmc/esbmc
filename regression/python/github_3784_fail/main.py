# Test dict.pop(key) on missing key raises KeyError (should fail)
d: dict[str, int] = {"a": 1}
d.pop("missing_key")
