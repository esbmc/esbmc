// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Pair of nested_mapping_in_contract_pass: the counter-claim
// `m[k][k] == v + 1` must be refuted because the read returns v.
contract A {
    mapping(address => mapping(address => uint256)) public m;
    function check(address k, uint v) public {
        m[k][k] = v;
        assert(m[k][k] == v + 1);
    }
}
