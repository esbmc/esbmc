// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// CORE FAILED soundness companion of mapping_struct_smtsort_k2_pass.
// Writes slot m[i][j].a, then (with i != j) asserts the transposed
// slot m[j][i].a equals the written value.  The two index dimensions
// are independent, so the transposed read must be violable: proves the
// K-deep native select/store chain keeps the dimension order (no
// i<->j aliasing in the struct-of-arrays decomposition).
contract C {
    struct S {
        uint256 a;
        uint256 b;
    }

    mapping(uint256 => mapping(uint256 => S)) m;

    function check(uint256 i, uint256 j, uint256 v) public {
        require(i != j);
        m[i][j].a = v;
        assert(m[j][i].a == v);
    }
}
