// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StorageContract {
    uint public value;

    function setValue(uint _value) public {
        value = _value;
    }
}

contract UserContract {
    StorageContract public s1;
    StorageContract public s2;

    constructor(address storageAddr) {
        require(storageAddr != address(0), "Invalid address");
        s1 = StorageContract(storageAddr);
        s2 =StorageContract(storageAddr); 
    }
    function updateAndAssert() public {
        s1.setValue(100);
        s2.setValue(200);
        assert(s1.value() == 200); 
    }
}