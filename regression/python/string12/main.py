def duplicate(x: str) -> str:
    return x


# Direct call
assert duplicate("direct") == "direct"

# Stored call
stored = duplicate("stored")
assert stored == "stored"

# Mixed comparison
assert duplicate("same") == stored if stored == "same" else True
