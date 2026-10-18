// SPDX-License-Identifier: MIT
// Dual of modifier_11: with the placeholder-nested-in-else fix applied,
// the modifier correctly inlines the wrapped body, so the `assert(false)`
// inside is actually reached and ESBMC reports a violation. Before the
// fix, the body was dropped and this test would (wrongly) pass.
pragma solidity >=0.8.0;

contract T {
    address public owner;

    modifier onlyOwner {
        if (msg.sender != owner) {
            revert();
        } else {
            _;
        }
    }

    constructor() {
        owner = msg.sender;
    }

    function detonate() public onlyOwner {
        assert(false);
    }
}
