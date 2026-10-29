// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// B5-B Stage S0 — same as cross_iter_internal_revert_leak but the
// mutated state is a mapping. Internal helpers go through legacy
// __ESBMC_assume(false) — no B1 rollback. The require() bypass on
// users[u] sets the pre-call mapping value to 0 (real Solidity
// default; ESBMC's mapping init is nondet).
contract H {
    mapping(address => uint256) public users;

    function _helper(address u) internal {
        users[u] = 1;
        require(false, "revert");
    }

    function tryWrite(address u) external {
        _helper(u);
    }

    function check(address u) external view {
        require(users[u] == 0);
        assert(users[u] == 0);
    }
}
