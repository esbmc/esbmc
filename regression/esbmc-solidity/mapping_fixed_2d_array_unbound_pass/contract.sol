// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// Round-trip test for `mapping(K => uint256[M][N])` in default unbound
// mode (no --bound, no `new Store()`).
//
// Built on the mapping(K => T[N]) routing fix (Phase 3 Fix B): the
// decl layer rewrites the state var to `mapping_t` and the helper
// `map_fixed_arr_get(&m, key, sizeof(T[M][N]))` lazily allocates the
// 2D slab. `m[k]` casts the void* return to `pointer<T[M]>` and the
// downstream `[i][j]` index expression then strides through the slab.
//
// KNOWNBUG (performance): same shape as mapping_fixed_array_unbound_pass
// — the helper path is sound but `map_get_raw` walks a linked list on
// every get/set, which doesn't solve within the regression timeout
// under k-induction × 6 writes + 6 reads.
// The sibling mapping_fixed_2d_array_unbound_fail runs to FAILED
// quickly (single write + single assertion), confirming the encoding
// is correct.
contract MappingFixed2DArrayUnboundPass {
    mapping(address => uint256[2][3]) public m;

    function check(address k) external {
        m[k][0][0] = 10;
        m[k][0][1] = 11;
        m[k][1][0] = 20;
        m[k][1][1] = 21;
        m[k][2][0] = 30;
        m[k][2][1] = 31;
        assert(m[k][0][0] == 10);
        assert(m[k][0][1] == 11);
        assert(m[k][1][0] == 20);
        assert(m[k][1][1] == 21);
        assert(m[k][2][0] == 30);
        assert(m[k][2][1] == 31);
    }
}
