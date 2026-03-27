def complex_control_flow(x: int, y: int, z: int) -> int:
    try:
        if x > 0:
            if y > 0:
                if z > 0:
                    return 1
                else:
                    pass  # Missing return here
            elif y < 0:
                return 2
            # Missing return when y == 0
        elif x < 0:
            return 3
        # Missing return when x == 0
    except Exception:
        return -2

# Test with values that should hit missing return paths
result = complex_control_flow(1, 0, 5)  # y == 0 case
