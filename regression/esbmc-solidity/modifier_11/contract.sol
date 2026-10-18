// SPDX-License-Identifier: MIT
// Regression for: modifier body with `_;` placeholder nested inside an
// if/else is correctly inlined — not silently dropped. Prior to the fix,
// the placeholder-splicing walker only scanned the modifier body's
// top-level statements, so `_;` nested inside a conditional was lost and
// the wrapped function's body never executed.
pragma solidity >=0.8.0;

contract T {
    uint256 public x;
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

    function setX(uint256 _v) public onlyOwner {
        x = _v;
        assert(x == _v);
    }
}
