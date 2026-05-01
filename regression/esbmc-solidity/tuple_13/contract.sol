// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Phase 4 basic: low-level call tuple destructuring with omission
contract LLCTuple {
    function test() public {
        (bool success, ) = msg.sender.call("");
        // success is nondet — both branches reachable
        if (success) {
            assert(true);
        }
    }
}
