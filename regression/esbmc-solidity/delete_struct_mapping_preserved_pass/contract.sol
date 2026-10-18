// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Solidity spec: `delete struct` resets non-mapping members and recurses
// into nested members EXCEPT mappings — mapping data must be preserved.
//
// Currently passes (the `_ESBMC_Mapping` placeholder field is just a
// 1-byte marker; real mapping data lives in the global
// `_ESBMC_map_storage` keyed by `(mid, addr, key)` where `mid` is a
// compile-time-constant linker address, unaffected by gen_zero).
//
// CORE: regression-protect this accidental correctness against the
// upcoming emit_delete_block refactor (which makes the mapping-skip
// explicit rather than incidental).
contract C {
    struct S {
        uint x;
        mapping(uint => uint) m;
    }
    S s;

    function f() public {
        require(s.x == 0 && s.m[7] == 0);
        s.x = 99;
        s.m[7] = 42;
        delete s;
        assert(s.x == 0);
        assert(s.m[7] == 42);
    }
}
