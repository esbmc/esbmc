// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression: in-contract nested mapping read-after-write must hold.
// `m[k][k] = v` writes via the combined-key path; the subsequent
// `m[k][k]` read uses the same combined key on the same backing
// mapping_t entry, so the assertion always succeeds.
contract A {
    mapping(address => mapping(address => uint256)) public m;
    function check(address k, uint v) public {
        m[k][k] = v;
        assert(m[k][k] == v);
    }
}
