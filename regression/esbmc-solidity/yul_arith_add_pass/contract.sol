// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    function check() public pure {
        uint a = 3;
        uint b = 4;
        uint r;
        assembly {
            r := add(a, b)
        }
        assert(r == 7);
    }
}
