def used() -> int:
    return 1


match used():
    case 1:
        assert used() == 1
