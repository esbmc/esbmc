// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// Regression test: transfer to dynamic contract instance with no receive/fallback.
//
// ROOT CAUSE (2026-06-24): see transfer_dynamic_self_pass_knownbug.
//
// This variant uses assert(false) to cleanly test path reachability.
//
// EXPECTED (correct) behaviour: `payable(address(c)).transfer(v)` where C has
//   no payable receive/fallback REVERTS (real EVM).  ESBMC should prune the
//   post-transfer path with __ESBMC_assume(false).  The assert(false) is then
//   unreachable → VERIFICATION SUCCESSFUL.
//
// CURRENT (buggy) behaviour: ESBMC lets the transfer "succeed" (EOA-fallback
//   debit, mislanded credit), so assert(false) IS reachable → VERIFICATION FAILED.

contract C {
    // Deliberately no receive() or fallback() — ETH transfer to this address
    // must revert on the real EVM.
    constructor() payable {}
    function send_to(address to, uint256 amount) public {
        payable(to).transfer(amount);
    }
}

contract InvMutTest {
    C c;
    constructor() payable {
        c = new C{value: 1 ether}();
    }
    // Probe: after a transfer to c (which has no receive), the post-transfer
    // code should be unreachable.  assert(false) acts as an "I am reachable" marker.
    function body() public {
        c.send_to(address(c), 1 ether);  // self-transfer, C has no receive → must revert
        // If ESBMC correctly prunes the reverted path: unreachable → vacuously true.
        // If ESBMC lets the path continue (bug): assert(false) fires → FAILED.
        assert(false);
    }
}
