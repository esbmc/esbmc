// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T2.2 — positive regression coverage for try/catch state rollback
// (struct-field case). Asserts that when target.mutateThenRevert
// reverts, the catch arm sees pre-call state (target.state() == 0).
// Already covered by the existing model: catch sits in the else branch
// of the nondet-split lowering at solidity_convert_stmt.cpp::
// TryStatement, so SSA branching keeps the success-arm's `state = v`
// write out of catch's view; on top of that, B1's per-frame snapshot
// in build_revert_rollback_block restores target's struct on the
// revert path. Either mechanism alone is sufficient. This file
// guards against future regressions in either.
contract Target {
    uint256 public state;
    function mutateThenRevert(uint256 v) external {
        state = v;
        revert("nope");
    }
}

contract H {
    Target target;

    constructor() {
        target = new Target();
    }

    function check(uint256 v) external {
        require(v != 0);
        try target.mutateThenRevert(v) {
            // unreachable: mutateThenRevert always reverts
        } catch {
            // catch should observe pre-call state.
            assert(target.state() == 0);
        }
    }
}
