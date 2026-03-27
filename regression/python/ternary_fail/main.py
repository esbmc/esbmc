def test_type_mismatch_failures() -> None:
    # This should fail - wrong expected value
    age: int = 18
    status: int = 1 if age >= 18 else 0
    assert status == 0  # WRONG: should be 1 since 18 >= 18


def test_logic_failures() -> None:
    # This should fail - incorrect condition evaluation
    x: float = 0.0
    positive: int = 1 if x > 0.0 else 0
    assert positive == 1  # WRONG: 0.0 is not > 0.0

    # Another logic error
    temperature: float = 98.0
    fever: int = 1 if temperature > 99.0 else 0
    assert fever == 1  # WRONG: 98.0 is not > 99.0


def test_float_precision_failures() -> None:
    # This should fail - wrong precision expectation
    is_enabled: bool = True
    power: float = 1.0 if is_enabled else 0.0
    assert power == 1.1  # WRONG: should be 1.0

    # Another precision error
    ratio: float = 0.5 if True else 0.25
    assert ratio == 0.75  # WRONG: should be 0.5


def test_boolean_failures() -> None:
    flag1: bool = True
    flag2: bool = False
    result: bool = True if (flag1 and flag2) else False
    assert result == True  # WRONG: True and False = False


test_type_mismatch_failures()
test_logic_failures()
test_float_precision_failures()
test_boolean_failures()
