// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// Regression test for cross-instance transfer between two dynamically allocated
// contracts of the same type.
//
// ROOT CAUSE (2026-06-24): see transfer_dynamic_self_pass_knownbug for full
//   explanation.  Same bug, different shape: c1 transfers 1 ether TO c2 (also
//   a dynamic C instance).  Because c2's address ≠ _ESBMC_Object_C.$address,
//   the dispatch loop misses, and the credit lands in sol_eoa_balance_array
//   instead of c2's $balance field.
//
// EXPECTED (correct) behaviour: after `c1.migrate(address(c2))`,
//   address(c2).balance == 3 ether (2 ether initial + 1 ether received).
//   The assertion `address(c2).balance == 2 ether` (unchanged) should be
//   FALSIFIED → VERIFICATION FAILED.
//
// CURRENT (buggy) behaviour: credit lands in EOA map, c2.$balance stays at
//   2 ether, so the wrong assertion vacuously passes → VERIFICATION SUCCESSFUL.

contract C {
    receive() external payable {}    // C2 CAN accept ETH — transfer does not revert.
    constructor() payable {}
    function migrate(address to) public {
        payable(to).transfer(address(this).balance);
    }
}

contract Probe {
    C c1;
    C c2;
    constructor() payable {
        c1 = new C{value: 1 ether}();
        c2 = new C{value: 2 ether}();
    }
    function check() public {
        // c1 sends its full 1 ether to c2.  After this c2 should hold 3 ether.
        c1.migrate(address(c2));
        // Correct: c2.balance == 3 ether → assertion (== 2) is FALSE → FAILED.
        // Bug:     c2.balance == 2 ether (credit lost) → assertion (== 2) is TRUE → SUCCESSFUL (wrong).
        assert(address(c2).balance == 2 ether);
    }
}
