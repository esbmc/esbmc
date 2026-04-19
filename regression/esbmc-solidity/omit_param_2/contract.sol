
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Example {

    function setValue(uint256 x, uint256) internal {
        // Implementation of setValue
    }

    function getValue() external{
        setValue(0,setValue2());
    }
    function setValue2() internal returns(uint){
        assert (1 == 0);
        return 0;
    }
}