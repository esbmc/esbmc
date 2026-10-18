// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// B3 fail dual — verifies the drain is actually observed.  The
// contract's `$balance` field MUST be zero after destroy()
// returns; an assertion that `this.balance` is still nonzero
// must FAIL for any feasible call sequence.  Pre-B3 this dual
// would also fail (path pruned by exit(0)) — but for the wrong
// reason.  Post-B3 it fails because the drain ran, which is the
// actual property we want to test.
contract Vault {
    constructor() payable {}

    function destroy(address payable to) public {
        selfdestruct(to);
    }

    function check() public view {
        // Should fail when destroy was called (this.balance == 0).
        assert(address(this).balance != 0);
    }
}
