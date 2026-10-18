// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
// (lhs_tuple) = cond ? (tuple_a) : (tuple_b) used to be silently dropped
// by construct_tuple_assigments (ERROR: cannot locate function call in RHS),
// leaving x/y at default zero and making BMC unsoundly "succeed". The fix
// decomposes the conditional element-wise into per-slot ternaries.
contract C {
    function check(uint64 a, uint64 b, bool c) external pure {
        uint64 x;
        uint64 y;
        (x, y) = c ? (a, b) : (b, a);
        if (c) {
            assert(x == a);
            assert(y == b);
        } else {
            assert(x == b);
            assert(y == a);
        }
    }
}
