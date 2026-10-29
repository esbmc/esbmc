// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// Round-trip test for `mapping(K => uint256[N])` in default unbound
// mode (no --bound, no `new Store()`).
//
// Semantic correctness: fixed by Phase 3 Fix B — the decl layer now
// rewrites `mapping(K => T[N])` state vars to `mapping_t` even under
// the !is_new_expr "static singleton" optimisation, and access routes
// through `map_fixed_arr_get`. The `array<T[N], inf>` sort that
// `src/solvers/smt/array_conv.cpp:92-95` could not encode is no
// longer produced.
//
// KNOWNBUG (performance): the helper-backed path is sound but slow —
// `map_get_raw` walks a linked list on every get/set, which bloats
// under k-induction × 3 writes + 3 reads. Bitwuzla accepts the
// encoding but does not solve within the regression timeout.
// Potential remedy: the `mapping_t_fast` variant (currently unused
// for non-scalar value types) or a dedicated fixed-arr keyspace
// encoder without the linked-list walk.
// See docs/claude/solidity/language-support.md §F.3.
contract MappingFixedArrayUnboundPass {
    mapping(address => uint256[3]) public m;

    function check(address k) external {
        m[k][0] = 10;
        m[k][1] = 20;
        m[k][2] = 30;
        assert(m[k][0] == 10);
        assert(m[k][1] == 20);
        assert(m[k][2] == 30);
    }
}
