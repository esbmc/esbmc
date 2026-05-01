// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Test: storage parameter modification propagates to caller (PASS)

struct Wrapper {
    uint256 value;
}

library TestLibrary {
    function setViaParam(Wrapper storage wrapper) public {
        wrapper.value = 10;
    }
}

contract TestContract {
    Wrapper wrapper;

    function test_storage_param() public {
        TestLibrary.setViaParam(wrapper);
        assert(wrapper.value == 10);
    }
}
