// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Test: using-for library call with storage parameter (PASS)

struct Wrapper {
    uint256 value;
}

library TestLibrary {
    function setTo42(Wrapper storage wrapper) public {
        wrapper.value = 42;
    }
}

contract TestContract {
    using TestLibrary for Wrapper;

    Wrapper wrapper;

    function test_using_for() public {
        wrapper.setTo42();
        assert(wrapper.value == 42);
    }
}
