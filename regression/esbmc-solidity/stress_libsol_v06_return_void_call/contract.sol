// SPDX-License-Identifier: MIT
pragma solidity ^0.6.0;

library L {
    function inner(uint256 a) internal pure {
        assert(a == a);
    }

    function wrap(uint256 a) internal pure {
        return inner(a);
    }
}

contract T {
    function f(uint256 a) public pure returns (uint256) {
        L.wrap(a);
        return a;
    }
}
