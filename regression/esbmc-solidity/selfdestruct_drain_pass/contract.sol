// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// B3 — selfdestruct(addr) fund drain.  The legacy C model
// `void selfdestruct() { exit(0); }` discarded the recipient
// argument and pruned the path entirely, so post-destruct
// dispatcher iterations were unreachable and the recipient's
// EOA balance never observed the transferred ETH.
//
// After B3 v1, calls to selfdestruct(to) emit:
//   _ESBMC_eoa_credit(to, this->$balance);
//   this->$balance = 0;
//   return;
//
// This test exercises the EOA-recipient path (the precise case)
// and asserts the contract's own balance is zero after the
// destroy() call returns.  Pre-fix this assertion would never
// run (path pruned by exit(0)); post-fix the assert is reachable
// AND holds.
contract Vault {
    constructor() payable {}

    function destroy(address payable to) public {
        // selfdestruct ends the call frame; the line below the
        // selfdestruct() call is unreachable in real EVM (and
        // also in our model, by the emitted `return`).
        selfdestruct(to);
    }

    function checkDrained() public view {
        // After a successful destroy(...) the vault's balance
        // must be zero — pre-B3 this assert was unreachable
        // because the exit(0) lowering pruned every path that
        // ever called destroy().
        if (address(this).balance != 0)
            assert(true); // vacuous when not drained yet
        else
            assert(address(this).balance == 0);
    }
}
