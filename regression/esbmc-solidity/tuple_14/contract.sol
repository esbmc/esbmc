// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Phase 4 adversarial: tuple from function with wrong value assertion
contract TupleLLCFail {
    function getPairAndFlag() public pure returns (bool, uint) {
        return (true, 42);
    }

    function test() public pure {
        (bool flag, ) = getPairAndFlag();
        // flag is true, but claim it's false
        assert(!flag);
    }
}
