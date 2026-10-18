// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Audit finding F2: xor_fold_key_to_64bit (solidity_convert_mapping.cpp:411)
// folds a 256-bit mapping key down to 64 bits via XOR. The "collision rate
// 2^-64" comment is a probability argument that is formally void in BMC/SMT
// — the solver can pick two distinct 256-bit keys whose 64-bit fold is the
// same, causing m[k2] = 200 to alias m[k1] in the SMT array.
//
// Closure requires 256-bit array domains across solvers (Bitwuzla / CVC5
// performance trade-off acknowledged). Hard fix; KNOWNBUG-locked.
contract H {
    mapping(uint256 => uint256) public m;
    function check(uint256 k1, uint256 k2) public {
        require(k1 != k2);
        m[k1] = 100;
        m[k2] = 200;
        assert(m[k1] == 100);
    }
}
