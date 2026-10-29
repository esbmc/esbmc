// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    function check() public pure {
        uint zero = 0;
        uint nonzero = 42;
        uint a;
        uint b;
        assembly {
            a := iszero(zero)
            b := iszero(nonzero)
        }
        assert(a == 1);
        assert(b == 0);
    }
}
