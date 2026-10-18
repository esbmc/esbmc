// SPDX-License-Identifier: MIT
pragma solidity ^0.6.0;

library L {
    function g() internal pure returns (bool) {
        return true;
    }

    function h() internal pure returns (uint256) {
        return 1;
    }
}

contract T {
    function f() public pure returns (uint256) {
        assert(L.g());
        assert(L.h() == 1);
        return L.h();
    }
}
