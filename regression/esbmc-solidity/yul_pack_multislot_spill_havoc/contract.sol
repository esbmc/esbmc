// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// OVER-REACH GUARD: `P{uint256 a; uint8 b}` does NOT fit one 32-byte slot
// (a fills slot 0, b spills to slot 1). The single-slot precise lowering MUST
// abort and fall back to havoc rather than silently mis-pack b into slot 0.
// We assert the `[approx]` havoc-fallback warning fires for this block; if a
// future change makes the precise path (incorrectly) accept a spilling struct,
// the warning vanishes and this test fails — catching a silent mis-lowering.
contract H {
    struct P {
        uint256 a;
        uint8 b;
    }
    P s;

    function check(uint256 va) public {
        assembly {
            sstore(s.slot, va)
        }
        uint256 ra;
        assembly {
            ra := sload(s.slot)
        }
        assert(ra == va);
    }
}
