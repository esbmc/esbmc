// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StorageContract {
    uint public value;

    function setValue(uint _value) public {
        value = _value;
    }
}

contract UserContract is StorageContract {
    function updateAndAssert() public {
        StorageContract.setValue(100);
        assert(value == 200);
    }
}
