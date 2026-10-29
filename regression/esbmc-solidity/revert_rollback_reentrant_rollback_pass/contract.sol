// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Codex review of the finding-#2 fix, P1: the global-store snapshot must be
// per-frame (like _sol_save_this), NOT a shared static slot.  Here f is
// re-entered via this.f(): the outer frame writes m[k]=1, the inner frame
// writes m[k]=2 then reverts, then the outer frame reverts.
//
// With a shared static snapshot, the inner entry overwrites the outer's
// snapshot, so the outer revert restores m[k] to the inner snapshot (1)
// instead of the true entry value (0) -> spurious VERIFICATION FAILED.
// With a per-frame snapshot each revert restores its own entry state, so
// m[k] ends at 0 and checkClean holds -> VERIFICATION SUCCESSFUL.
contract C {
    mapping(uint => uint) m;

    function f(uint k, uint depth) public {
        m[k] = depth + 1;
        if (depth == 0)
            this.f(k, 1);
        revert();
    }

    function checkClean(uint k) public view {
        assert(m[k] == 0);
    }
}
