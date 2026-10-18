// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    function check() public pure {
        uint a = 11;
        uint b = 22;
        uint result;
        assembly {
            result := a
            if gt(b, a) {
                result := b
            }
        }
        assert(result == 22);
    }
}
