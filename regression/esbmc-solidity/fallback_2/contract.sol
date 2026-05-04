// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Test fallback() function body verification.
// The assertion inside fallback should fail because msg.sender
// is nondeterministic and not necessarily address(0).
contract FallbackFail {
    fallback() external {
        // FAILS: msg.sender is nondet
        assert(msg.sender == address(0));
    }
}
