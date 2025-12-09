pragma solidity >=0.5.0;

contract Base {
    FreshJuiceSize choice;

    function setLarge() public {
        choice = FreshJuiceSize.LARGE;
        uint x = getChoice();
        assert(x == 2);
        assert(uint(FreshJuiceSize.MEDIUM) == 1);
    }

    function getChoice() public view returns (uint) {
        return uint(choice);
    }
}

enum FreshJuiceSize {
    SMALL,
    MEDIUM,
    LARGE
}
