// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression for the `phi_function: no symbol for ...` crash on
// `<library-fn> storage ref = self.<field>` aliases (1inch
// liquidity-protocol's ExplicitLiquidVoting._update). The storage-ref
// alias mechanism used to flatten a MemberAccess RHS into a bare symbol
// referencing the struct field with a function-local naming template,
// which never landed in the symbol table — goto-symex then warned and
// dropped the assignment. This contract exercises the pattern (storage
// ref to `self.data` inside an inner block scope under a guard so the
// phi_function path runs) and asserts a property that must hold after
// the alias is correctly resolved to a member access on `self`.

library V {
    struct Data { uint104 result; }
}

library L {
    struct Data { V.Data data; uint256 sum; }

    function _update(Data storage self, uint256 v) internal {
        if (v != 0) self.sum = v;
        {
            V.Data memory snap = self.data;
            if (snap.result != 0) {
                V.Data storage sd = self.data;
                sd.result = uint104(v);
            }
        }
    }
}

contract C {
    L.Data d;
    function go(uint256 v) public {
        L._update(d, v);
        // After _update, sum tracks v exactly when v != 0.
        if (v != 0) assert(d.sum == v);
    }
}
