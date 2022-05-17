pragma solidity >=0.7.0 <0.9.0;

contract StateVarInit {
    uint8 x = 3;

    function state_var_init() public {
        assert(x == 4);
    }
}
