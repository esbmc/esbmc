// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    function check() public pure {
        uint r;
        assembly {
            let x := 10
            let y := 20
            let z := add(x, y)
            r := z
        }
        assert(r == 30);
    }
}
