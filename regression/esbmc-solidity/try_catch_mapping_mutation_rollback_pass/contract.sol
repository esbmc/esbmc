// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T2.2 — positive regression coverage for try/catch state rollback
// (mapping case). Asserts that when target.put writes a mapping then
// reverts, the catch arm sees the pre-call mapping value.
// Already covered by the if/else nondet-split lowering: catch is in
// the else branch and the put's `users[u] = v` write is predicated on
// the then-branch, so SSA at the else-branch read of `target.users(u)`
// resolves to the pre-call value. The explicit `require(target.users(u)
// == 0)` precondition is needed because ESBMC's mapping model starts
// from a nondet baseline (real Solidity defaults to 0 — pre-existing
// gap, separate from try/catch rollback).
contract Target {
    mapping(address => uint256) public users;

    function put(address u, uint256 v) external {
        users[u] = v;
        revert("nope");
    }
}

contract H {
    Target target;

    constructor() {
        target = new Target();
    }

    function check(address u, uint256 v) external {
        require(v != 0);
        require(target.users(u) == 0);  // mapping starts at 0 (real Solidity)
        try target.put(u, v) {
            // unreachable
        } catch {
            assert(target.users(u) == 0);
        }
    }
}
