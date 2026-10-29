// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

// Stage 3 mapping refactor exhibitor: four DISTINCT mapping state vars
// each touched at the same key.  Verifies that writes to one mapping at
// key K do not alias to reads of another mapping at the same key K — i.e.
// per-state-var keyspace separation through the explicit `mapping_t.mid`
// field.
//
// Pre-Stage-2 the linked list keyed lookups by (addr, key) only; with
// every mapping in the same contract sharing an `addr`, two different
// mapping fields at the same key would have walked into each other's
// list and the only thing distinguishing them was that each field had
// its own `_ESBMC_Mapping*` head pointer.  Stage 2 collapsed every
// mapping into one global SMT array, using the per-state-var
// `_ESBMC_inf_<name>` symbol address as a `mid` discriminator (cast
// (uintptr_t)m->base).  Stage 3 makes the discriminator a real uint64
// field on `mapping_t`, populated by the frontend's
// `next_mapping_mid` counter.
//
// Concretely: m1[42] = 1; m2[42] = 2; m3[42] = 3; m4[42] = 4.  All four
// must round-trip independently.  If any two mid values collide, the
// solver finds a counterexample where some assertion fails.
contract MapSmtArrayMultiStateVar {
    mapping(uint256 => uint256) public m1;
    mapping(uint256 => uint256) public m2;
    mapping(uint256 => uint256) public m3;
    mapping(uint256 => uint256) public m4;

    function check() external {
        m1[42] = 1;
        m2[42] = 2;
        m3[42] = 3;
        m4[42] = 4;

        assert(m1[42] == 1);
        assert(m2[42] == 2);
        assert(m3[42] == 3);
        assert(m4[42] == 4);

        // Cross-mapping non-aliasing: writing to one must not change
        // another.  Different `mid` discriminators give disjoint slot
        // indices in the global SMT array.
        m1[42] = 999;
        assert(m2[42] == 2);
        assert(m3[42] == 3);
        assert(m4[42] == 4);
        assert(m1[42] == 999);
    }
}
