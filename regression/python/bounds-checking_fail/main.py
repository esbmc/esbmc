def access_list_element(index: int) -> int:
    my_list = [10, 20, 30, 40, 50]
    return my_list[index]

def test_bounds() -> None:
    # Valid access
    assert access_list_element(2) == 30
    
    # This will trigger a bounds violation
    result = access_list_element(10)  # Index out of bounds
    assert result == 0

test_bounds()
