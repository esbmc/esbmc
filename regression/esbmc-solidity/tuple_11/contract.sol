// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Phase 2 basic: external call returning tuple via new
contract Provider {
    function getPair() public pure returns (uint, uint) {
        return (10, 20);
    }
}

contract Consumer {
    function test() external {
        Provider p = new Provider();
        (uint a, uint b) = p.getPair();
        assert(a == 10);
        assert(b == 20);
    }
}
