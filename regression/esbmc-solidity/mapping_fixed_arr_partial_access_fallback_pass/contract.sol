// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// Gate-coverage test: a partial mapping access (`m[k]` returned as a
// memory copy) cannot be encoded by the flat-array encoder because
// there is no slab pointer to copy from.  The decl-time AST scanner
// `has_partial_mapping_access` detects access depth < 1+nesting and
// falls back to the slow `mapping_t + map_fixed_arr_get` path.
//
// `tmp` here is a Solidity memory copy of the entire `m[k]` slab.
// Writes to `tmp[i]` are local — they do not flow back to `m`.
// Subsequent `tmp[i]` reads must see the values written into `tmp`.
//
// Under the slow path this works because `m[k]` returns a slab
// pointer and the memory copy is a real heap allocation.  Under the
// flat encoder there's no slab — partial access would be unsound, so
// the gate must fall back here.
contract MappingFixedArrPartialAccessFallbackPass {
    mapping(address => uint256[3]) m;

    function check(address k) external view {
        uint256[3] memory tmp = m[k];
        // tmp is a fresh memory copy initialised from m[k]'s
        // current values (all zero since never written).
        assert(tmp[0] == 0);
        assert(tmp[1] == 0);
        assert(tmp[2] == 0);
    }
}
