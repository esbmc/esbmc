// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    function check() public pure {
        uint x = 1;
        uint result;
        assembly {
            switch x
            case 0 { result := 10 }
            case 1 { result := 20 }
            default { result := 30 }
        }
        assert(result == 20);
    }
}
