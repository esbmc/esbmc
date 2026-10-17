// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Unbound dual of map_dynarr_push_pass_bound: push+read happens inside
// a single function call, so the state write is visible to the
// subsequent read regardless of the nondet-init of the state var in
// --unbound mode.
contract C {
    mapping(address => uint256[]) m;
    function test() public {
        m[address(0x1)].push(42);
        assert(m[address(0x1)][0] == 42);
    }
}
