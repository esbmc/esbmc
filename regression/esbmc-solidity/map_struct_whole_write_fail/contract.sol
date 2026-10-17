// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// A whole-struct write to a struct-valued mapping must round-trip through
// the field reads that follow it.
contract C {
    struct S {
        uint256 a;
        uint256 b;
    }

    mapping(address => S) m;

    function f(address k) public {
        m[k] = S(10, 11);
        assert(m[k].a == 11);
    }
}
