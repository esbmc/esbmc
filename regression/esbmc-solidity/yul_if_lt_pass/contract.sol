// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    function check() public pure {
        uint a = 7;
        uint b = 3;
        uint result;
        assembly {
            result := a
            if lt(b, a) {
                result := b
            }
        }
        assert(result == 3);
    }
}
