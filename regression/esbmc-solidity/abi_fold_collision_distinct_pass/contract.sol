// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Audit finding F1 (CLOSED): the legacy `_ESBMC_abi_fold` used a
// multiplicative fold `acc * 0x100000001b3 + next` mod 2^256, which is
// not a permutation under wraparound — the SMT solver could find
// (a, b) != (c, d) with equal fold output, breaking distinct-args →
// distinct-hashes for `keccak256(abi.encode(...))`.
//
// Closure (ledger #3): the multi-arg fold path was rewritten to use
// bit-vector concat into wide-BV-indexed tables (`_ESBMC_abi_table_<W>`,
// `_ESBMC_keccak_table_<W>`, etc., each annotated `__ESBMC_inf_size:<W>`).
// At each fold-path call site the frontend allocates a static-global
// (key, result) pair, looks up `<table>[concat]`, and emits per-callsite
// distinctness assumes `prior_key == this_key || prior_result != this_result`
// against every prior matching call site. The SMT array axiom gives the
// same-args→same-hash direction; the explicit per-pair assumes give
// the distinct-args→distinct-hashes direction (injectivity).
contract H {
    function check(uint256 a, uint256 b, uint256 c, uint256 d) public pure {
        require(!(a == c && b == d));
        assert(keccak256(abi.encode(a, b)) != keccak256(abi.encode(c, d)));
    }
}
