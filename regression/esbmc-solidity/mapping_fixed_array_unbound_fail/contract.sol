// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// Violation test for `mapping(K => uint256[N])` in default unbound mode.
// Under correct semantics the `m[k][0] != 10` assertion is violated.
//
// Fixed by Phase 3 Fix B: decl layer rewrites the mapping state var to
// `mapping_t` and routes access through `map_fixed_arr_get`, side-stepping
// the `array<uint256[N], inf>` array-of-array sort that
// `src/solvers/smt/array_conv.cpp:92-95` cannot encode.
// See docs/claude/solidity/language-support.md §F.3.
contract MappingFixedArrayUnboundFail {
    mapping(address => uint256[3]) public m;

    function check(address k) external {
        m[k][0] = 10;
        assert(m[k][0] != 10);
    }
}
