// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Inverted onlyOwner: non-owner passes (require msg.sender != owner).
// In-function assert(msg.sender == owner) MUST falsify because the
// require's ASSUME enforces msg.sender != owner along every reached
// path, contradicting the assertion. Detection requires per-tx
// ambient reseed (the `_sol_per_tx_reseed` mechanism in
// `_ESBMC_Main_Bug`'s while-loop) so msg.sender is fresh-nondet per
// dispatcher iteration; otherwise constructor's
// `owner = msg.sender = α` invariant freezes msg.sender == owner.
contract Bug {
    address public owner;
    constructor() { owner = msg.sender; }
    modifier onlyOwner() { require(msg.sender != owner); _; }
    function privileged() public onlyOwner {
        assert(msg.sender == owner);
    }
}
