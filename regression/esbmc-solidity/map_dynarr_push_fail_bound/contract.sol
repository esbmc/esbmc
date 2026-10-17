// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Fail-side dual of map_dynarr_push_pass_bound: the assertion claims
// the wrong value and must be refuted by symex.
contract C {
    mapping(address => uint256[]) m;
    function test() public {
        m[address(0x1)].push(42);
        assert(m[address(0x1)][0] != 42);
    }
}
