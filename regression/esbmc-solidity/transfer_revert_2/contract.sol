// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Bound-mode transfer revert semantics:
// Real Solidity transfer() reverts on insufficient balance. Any code
// immediately after the transfer is unreachable on the insufficient
// balance path. The old model returned `false` and let callers keep
// running after a failed transfer; with revert semantics
// (__ESBMC_assume(false) in the helper), the failing path is pruned
// and an `assert(false)` placed right after the transfer is vacuously
// true — i.e., verification passes because the program can never
// reach that point.
contract Receiver {
    receive() external payable {}
}

contract TransferRevert {
    Receiver r;

    function __ESBMC_assume(bool) internal pure {}

    constructor() {
        r = new Receiver();
    }

    function test() public {
        __ESBMC_assume(address(this).balance == 5);
        payable(address(r)).transfer(10); // insufficient: must revert
        // Unreachable under correct revert semantics.
        assert(false);
    }
}
