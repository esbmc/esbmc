// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Test: named return parameter — implicit return (VERIFICATION SUCCESSFUL)
contract NamedReturnPass {
    function add(uint x, uint y) public pure returns (uint result) {
        result = x + y;
    }

    function test(uint a, uint b) public pure {
        uint v = add(a, b);
        assert(v == a + b);
    }
}
