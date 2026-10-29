// SPDX-License-Identifier: MIT
// Sanity test for the splice_placeholders fix: a non-empty (N=1) body
// annotated by an `if (cond) _;` modifier must continue to expand to
// `if (cond) { body_stmt }` — the parent-aware substitution wraps
// body_exprt's operands in a code_blockt, but for N=1 the result is
// semantically identical to the legacy flat-splice form, just with
// one extra (transparent) block scope. Verifies that the wrap doesn't
// disturb scoping or symbol resolution for the common case.
pragma solidity >=0.8.0;

contract C {
    address owner;
    uint256 x;

    constructor() {
        owner = msg.sender;
        setX(7);
        // Post-call: owner is permitted, body ran, x == 7.
        assert(x == 7);
    }

    modifier g {
        if (msg.sender == owner) _;
    }

    function setX(uint256 v) public g {
        x = v;
    }
}
