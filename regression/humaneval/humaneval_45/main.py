# HumanEval/45
# Entry Point: triangle_area
# ESBMC-compatible format with direct assertions



def triangle_area(a, h):
    """Given length of a side and high return area for a triangle.
    >>> triangle_area(5, 3)
    7.5
    """
    return a * h / 2.0

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert triangle_area(5, 3) == 7.5
    # assert triangle_area(2, 2) == 2.0
    # assert triangle_area(10, 8) == 40.0
    print("✅ HumanEval/45 - All assertions completed!")
