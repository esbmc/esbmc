// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// CORE FAILED soundness companion of mapping_struct_smtsort_k2_pass.
// Same K=2 array^K<struct> write, but the post-condition is the
// negation (a == v + 1).  Proves the M3 struct-of-arrays round-trip is
// not vacuous: the solver must find the violating model.
contract C {
    struct S {
        uint256 a;
        uint256 b;
    }

    mapping(uint256 => mapping(uint256 => S)) m;

    function check(uint256 i, uint256 j, uint256 v) public {
        m[i][j].a = v;
        assert(m[i][j].a == v + 1);
    }
}
