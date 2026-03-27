def test_sorted_basic():
    l = [3, 2, 1]
    result = sorted(l)
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    # Original unchanged
    assert l[0] == 3
    assert l[1] == 2
    assert l[2] == 1

def test_sorted_already_sorted():
    l = [1, 2, 3, 4, 5]
    result = sorted(l)
    assert result[0] == 1
    assert result[4] == 5

def test_sorted_reverse():
    l = [5, 4, 3, 2, 1]
    result = sorted(l)
    assert result[0] == 1
    assert result[4] == 5

def test_sorted_duplicates():
    l = [3, 1, 2, 1, 3]
    result = sorted(l)
    assert result[0] == 1
    assert result[1] == 1
    assert result[2] == 2
    assert result[3] == 3
    assert result[4] == 3

def test_sorted_single():
    l = [42]
    result = sorted(l)
    assert result[0] == 42
    assert l[0] == 42

test_sorted_basic()
test_sorted_already_sorted()
test_sorted_reverse()
test_sorted_duplicates()
test_sorted_single()
