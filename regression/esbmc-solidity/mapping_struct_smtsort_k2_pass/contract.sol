// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// CORE (non-coverage / plain verification).  Was KNOWNBUG until Stage
// 2C.2d; flipped CORE by the M3 struct-of-arrays fix.
//
// State-var nested mapping whose value is a struct =
// array^K<Struct{...}> with K>=2 and infinite outer levels.  The
// default node tuple flattener (bitwuzla) used to convert_sort an
// array-of-array-of-struct and hit NW1 (smt_conv.cpp's array_id case ->
// to_solver_smt_sort<> on a bare struct sort: "bare smt_sort (id=4)").
//
// Stage 2C.2a-2d represent a K>=2 array-of-struct as a struct-of-arrays
// tuple_node over solver-native per-field arrays; select/update/assign
// distribute per field.  The write/read round-trip now holds.
//
// Dual: mapping_struct_smtsort_k1_pass (K=1, identical flags) is the
// K=1-byte-identical regression guard.  Soundness companions:
// mapping_struct_smtsort_k2_{dual,sibling,dimorder}_fail.
contract C {
    struct S {
        uint256 a;
        uint256 b;
    }

    mapping(uint256 => mapping(uint256 => S)) m;

    function check(uint256 i, uint256 j, uint256 v) public {
        m[i][j].a = v;
        assert(m[i][j].a == v);
    }
}
