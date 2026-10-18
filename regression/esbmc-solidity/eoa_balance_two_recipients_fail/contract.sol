// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Negative counterpart: two distinct EOAs receive amounts x and y.
// Asserting the resulting balances are equal is provably false for
// nondet x != y (the EOA balance map keeps them distinct).
contract Bal {
    constructor() payable {}

    function check(address payable a, address payable b, uint x, uint y) public {
        if (a == b) return;
        if (a == address(this) || b == address(this)) return;
        if (a.balance != 0 || b.balance != 0) return;
        if (address(this).balance < x) return;
        a.transfer(x);
        if (address(this).balance < y) return;
        b.transfer(y);
        // Wrong: a.balance == x and b.balance == y; equality only when x == y.
        assert(a.balance == b.balance);
    }
}
