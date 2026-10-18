// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Dual to stress_libsol_storage_ref_member_alias_pass: validates that
// after the storage-ref-to-MemberAccess fix, ESBMC still finds bugs on
// the same code shape. If goto-symex were silently dropping the
// assignment through the alias (the original `phi_function: no symbol`
// behavior), the violation below would be masked.

library V {
    struct Data { uint104 result; }
}

library L {
    struct Data { V.Data data; uint256 sum; }

    function _update(Data storage self, uint256 v) internal {
        if (v != 0) self.sum = v;
        {
            V.Data memory snap = self.data;
            if (snap.result != type(uint104).max) {
                V.Data storage sd = self.data;
                sd.result = uint104(v);
            }
        }
    }
}

contract C {
    L.Data d;
    function go(uint256 v) public {
        require(v != 0 && v <= type(uint104).max);
        L._update(d, v);
        // The alias assignment must have written through to d.data.result.
        // If the storage-ref alias were dropped (the original bug), this
        // would still hold (d.data.result stays 0) and the assertion below
        // would *fail to detect* the bug. With the fix in place, the write
        // is observed, so the negated property is violated and ESBMC
        // reports a counterexample — exactly what we want here.
        assert(d.data.result != uint104(v));
    }
}
