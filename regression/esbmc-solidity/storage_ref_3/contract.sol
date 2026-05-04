// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Test: local storage variable aliases the storage parameter (PASS)

struct Wrapper {
    uint256 value;
}

library TestLibrary {
    function setViaLocal(Wrapper storage wrapper) public {
        Wrapper storage secondWrapper = wrapper;
        secondWrapper.value = 20;
    }
}

contract TestContract {
    Wrapper wrapper;

    function test_storage_local() public {
        TestLibrary.setViaLocal(wrapper);
        assert(wrapper.value == 20);
    }
}
