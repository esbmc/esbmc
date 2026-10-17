// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Audit finding F3: _ESBMC_map_idx (solidity_mapping.c:70-87) folds
// (mid, addr, key) to a 64-bit slot index via XOR. The comment explicitly
// states "Collision rate 2^-64 per pair is acceptable per existing
// precedent" — a probability argument the SMT solver disregards. Two
// distinct (mid, addr, key) triples can fold to the same slot, causing
// false aliasing across mappings.
//
// This test exercises a state-variable mapping owned by a tracked
// contract instance (which routes through _ESBMC_map_idx for the
// (mid, this->$address, key) keying) — distinct from F2's frontend-
// fold path.
contract H {
    address public a1;
    address public a2;
    mapping(address => uint256) public m;
    constructor(address _a1, address _a2) {
        require(_a1 != _a2);
        a1 = _a1;
        a2 = _a2;
        m[a1] = 100;
        m[a2] = 200;
    }
    function check() public view {
        assert(m[a1] == 100);
    }
}
