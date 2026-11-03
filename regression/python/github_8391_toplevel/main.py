def used() -> int:
    return 1


match used():
    case int():
        assert used() == 1
