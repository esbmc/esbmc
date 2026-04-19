pragma solidity >=0.5.0;

contract Contract {
    function func() public {
        uint x = gasleft();
        uint y = gasleft();
        assert(x > y);
    }
}
