// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    function check() public pure {
        uint x = 21;
        uint result;
        assembly {
            let temp := mul(x, 2)
            result := temp
        }
        assert(result == 42);
    }
}
