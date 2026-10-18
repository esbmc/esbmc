// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Root-cause probe for SWC-115: without any constraint, the EVM
// state space contains call frames where `tx.origin` and `msg.sender`
// are distinct (contract-to-contract relay).  The dispatcher must
// pick `msg_sender` and `tx_origin` independently at each top-level
// invocation, so this assertion must FAIL.  Previously a stray
// `__ESBMC_assume(tx_origin == msg_sender)` in `_sol_per_tx_reseed`
// forced them equal and made this assertion vacuously pass — masking
// SWC-115 phishing patterns.
contract Equal {
    uint256 public x;
    function f(uint256 v) external {
        assert(msg.sender == tx.origin);
        x = v;
    }
}
