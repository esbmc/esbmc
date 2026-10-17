// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

// Companion fail dual to map_smt_array_multi_state_var_pass.
//
// The pass test asserts cross-mapping isolation: m1[42] = 999 must not
// change m2[42].  This fail test inverts the assertion and claims they
// are equal — the verifier should find a counterexample because the
// per-state-var `mid` discriminator gives disjoint slots in the global
// SMT array.  If the fail test were to PASS, that would be a soundness
// regression (mid collision aliasing) we want surfaced immediately.
contract MapSmtArrayMultiStateVarFail {
    mapping(uint256 => uint256) public m1;
    mapping(uint256 => uint256) public m2;

    function check() external {
        m1[42] = 1;
        m2[42] = 2;
        // Wrong: m1 and m2 are distinct mappings.  This should FAIL.
        assert(m1[42] == m2[42]);
    }
}
