// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    function check() public pure {
        uint y = 5;
        assembly {
            mstore(0x40, y)
        }
        assert(y == y);
    }
}
