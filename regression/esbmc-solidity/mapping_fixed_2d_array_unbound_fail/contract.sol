// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// Violation test for `mapping(K => uint256[M][N])` in default unbound mode.
// Under correct semantics the `m[k][0][0] != 10` assertion is violated.
contract MappingFixed2DArrayUnboundFail {
    mapping(address => uint256[2][3]) public m;

    function check(address k) external {
        m[k][0][0] = 10;
        assert(m[k][0][0] != 10);
    }
}
