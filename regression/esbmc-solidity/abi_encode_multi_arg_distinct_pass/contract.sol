// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T2.1 Stage S0 — KNOWNBUG pinning the multi-arg conflation in abi.encode.
// Real Solidity: changing the second arg changes the encoding bytes →
//                keccak hashes differ.
// Today: solidity_convert_expr.cpp:2395-2436 takes ONLY the first compatible
//        arg, so abi.encode(a, b) ≡ abi.encode(a, c) ≡ identity-on-a;
//        keccak(a) == keccak(a) → assert(!=) FAILS.
// After Stage S1 (multi-arg fold): fold(a, b) != fold(a, c), assertion
//        holds → SUCCESSFUL → flip to CORE / drop _knownbug suffix.
contract H {
    function check(uint256 a, uint256 b, uint256 c) external pure {
        require(b != c);
        assert(keccak256(abi.encode(a, b)) != keccak256(abi.encode(a, c)));
    }
}
