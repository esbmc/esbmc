data = b'ABC'
values = [65, 66, 66]  # ASCII values

for i in range(len(data)):
    assert data[i] == values[i]
