import pytest
def calc_final_price(original: float, is_vip: bool) -> float:
    if is_vip:
        return original * 0.9
    return original


@pytest.mark.parametrize(
    "original, is_vip, expected",
    [
        (100.0, False, 100.0),  
        (200.0, True, 180.0),  
        (300.0, True, 100.0),   
    ],
)
def test_calc_final_price(original: float, is_vip: bool, expected: float) -> None:
    result = calc_final_price(original, is_vip)
    assert result == expected

test_calc_final_price(150.0, True, 150)  
