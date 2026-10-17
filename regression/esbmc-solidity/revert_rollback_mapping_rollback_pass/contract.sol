// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Adversarial-review finding #2: mapping backing stores live in a global
// (`sol:@C@C@m`) keyed by `$address`, OUTSIDE the *this struct that the
// revert rollback restores.  A mapping write in a reverted transaction
// used to leak into the next transaction of the bounded multi-tx harness.
//
// Real EVM: setAndRevert reverts, so m[k] is rolled back to 0, and the
// assert in checkClean always holds -> VERIFICATION SUCCESSFUL.
// build_revert_rollback_block now snapshots the mapping global at entry and
// restores it alongside *this on revert, so the leak is gone.  Was KNOWNBUG
// (spurious VERIFICATION FAILED) before the fix.
contract C {
    mapping(uint => uint) m;

    function setAndRevert(uint k) public {
        m[k] = 1;
        revert();
    }

    function checkClean(uint k) public view {
        assert(m[k] == 0);
    }
}
