// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Phase 2 adversarial: wrong assertion on external call tuple return
contract Provider {
    function getPair() public pure returns (uint, uint) {
        return (10, 20);
    }
}

contract Consumer {
    function test() external {
        Provider p = new Provider();
        (uint a, uint b) = p.getPair();
        // a is 10, b is 20. Claim b == 10 (swapped — wrong).
        assert(b == 10);
    }
}
