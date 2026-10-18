// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// Gate-coverage test: a `T[N] storage ref = m[k]` aliases the slab and
// must NOT be encoded by the per-mapping flat-array encoder (the flat
// encoder doesn't materialise a slab, so writes through the alias
// would not flow back to subsequent `m[k][i]` reads).
//
// The decl-time AST scanner `has_mapping_storage_ref` detects this
// pattern and falls back to the slow `mapping_t + map_fixed_arr_get`
// helper path, where the slab pointer is real and the alias is sound.
//
// If the gate ever leaks (flat encoder fires here), the post-write
// read via the original mapping access would see 0 instead of 5 —
// the assertion would FAIL even though the source semantics says
// SUCCESSFUL.  This test is the soundness oracle.
contract MappingFixedArrStorageRefFallbackPass {
    mapping(address => uint256[3]) m;

    function check(address k) external {
        uint256[3] storage ref = m[k];
        ref[0] = 5;
        assert(m[k][0] == 5);
    }
}
