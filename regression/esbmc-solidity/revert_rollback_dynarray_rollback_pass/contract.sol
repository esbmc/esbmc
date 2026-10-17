// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Adversarial-review finding #2 (dynamic-array facet): a state-variable
// dynamic array's elements and length live in global stores keyed by
// `$address`, outside the `*this` struct that `revert` restores.  A
// `push` in a reverted transaction leaks its length increment into the
// next transaction.
//
// Real EVM: pushAndRevert reverts -> a.length stays 0 -> assert holds ->
// SUCCESSFUL.  Before the fix, a.length == 1 leaked -> spurious FAILED.
// build_revert_rollback_block now snapshots the `<arr>_dynarray_len` (and
// element-data) globals at entry and restores them on revert.  Was KNOWNBUG.
contract C {
    uint[] a;

    function pushAndRevert() public {
        a.push(7);
        revert();
    }

    function checkClean() public view {
        assert(a.length == 0);
    }
}
