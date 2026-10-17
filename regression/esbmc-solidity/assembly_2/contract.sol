// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Pre-T2.4 this asserted x == 6 under havoc-mode and was expected FAILED.
// Post-T2.4 the precise Yul lowerer deterministically computes x = 6, so
// the assertion holds and the test flipped to SUCCESSFUL. Contract name
// kept for git-history continuity; the new name would be AssemblyAdd.
contract AssemblyFail {
    function test() public pure {
        uint x = 5;
        assembly {
            x := add(x, 1)
        }
        assert(x == 6);
    }
}
