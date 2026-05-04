// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Adversarial: large keys store different values, assert wrong value
contract LargeKeyMappingFail {
    mapping(uint256 => uint256) public m;

    function test() public {
        uint256 k1 = 1;
        uint256 k2 = (1 << 128) + 1;

        m[k1] = 100;
        m[k2] = 200;

        // m[k2] is 200, not 100
        assert(m[k2] == 100);
    }
}
