// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// Default-zero semantics under the per-mapping flat-array encoder.
// Solidity guarantees that a never-written `mapping(K => T[N])[unwritten_k][i]`
// reads as zero.  The flat encoder achieves this via the
// `__ESBMC_inf_size:W` annotation on the per-mapping infinite array
// — ESBMC's main-entry zero-init makes `select(arr, idx) = 0` for
// any unwritten slot.
//
// If the encoder were unsound (e.g., uninitialised SMT array values
// were nondet), the assertion below would FAIL.
contract MappingFixedArrDefaultZeroPass {
    mapping(address => uint256[3]) m;

    function check(address k) external view {
        // Never written; expect zero.
        assert(m[k][0] == 0);
        assert(m[k][1] == 0);
        assert(m[k][2] == 0);
    }
}
