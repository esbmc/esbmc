// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Struct containing a fixed-size array — post-clone equality holds.
// This exercises two pieces that had to ship together:
//   1. Phase 2 ctor walker (emit_ctor_deep_init_fixup) recursively
//      calloc's nested pointer-backed storage.  For this test it emits
//      `this->bx.cells = _ESBMC_alloc_array(2, 32)` in C's ctor, so
//      `base->bx.cells` is a valid buffer before setCells() writes to
//      it — previously bx.cells stayed NULL and the write went to
//      nondet memory.
//   2. Phase 1 clone walker (emit_clone_deep_copy_fixup) recurses into
//      the inline `bx` struct, emits `_ESBMC_arrcpy(base->bx.cells, 2,
//      32)` for the fixed-array sub-field, and skips the nested
//      `struct Box {...}` TYPE DECLARATION that Solidity stores as a
//      component of the outer contract struct — that type-decl has
//      `type.id()=="struct"` and no storage slot, so treating it as a
//      field would generate a malformed `base->.cells` access.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    struct Box { uint256[2] cells; }
    Box private bx;
    function setCells(uint256 a, uint256 b) public {
        bx.cells[0] = a; bx.cells[1] = b;
    }
    function cell(uint256 i) public view returns (uint256) { return bx.cells[i]; }
}

contract H {
    function check(uint256 a, uint256 b) public {
        C base = new C();
        base.setCells(a, b);
        C clone = __ESOL_deep_copy(base);
        assert(clone.cell(0) == a);
        assert(clone.cell(1) == b);
    }
}
// KNOWNBUG: the inductive step cannot prove this without the inductive
// invariants the local k-induction inferred (loop-entry invariants,
// counted-loop skip, per-field havoc), which upstream's rewrite dropped.
