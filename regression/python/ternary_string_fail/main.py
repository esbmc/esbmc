def test_string_comparison_failure() -> None:
    user_role: str = "admin"
    access: str = "granted" if user_role == "admin" else "denied"
    assert access == "denied"  # WRONG: should be "granted"


def test_string_selection_failure() -> None:
    is_weekend: bool = False
    day_type: str = "weekend" if is_weekend else "weekday"
    assert day_type == "weekend"  # WRONG: should be "weekday"


test_string_comparison_failure()
test_string_selection_failure()
