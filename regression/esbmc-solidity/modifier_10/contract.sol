pragma solidity ^0.8.0;

contract Test {

    uint public x;

    function checkValue() internal view returns (bool) {
        return x == 1;
    }

    modifier onlyWhenXIsOne() {
        require(checkValue(), "x is not 1");
        _;
    }

    function setX(uint _x) public {
        x = _x;
    }

    function restrictedFunction() public onlyWhenXIsOne {
        assert(1==0);
    }
}
