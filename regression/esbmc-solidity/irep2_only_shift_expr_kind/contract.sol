// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// The converter emits >> without picking the shift kind, and IREP2 has no
// kind-less shift node, so the seam resolves it by the left operand's
// signedness. Each operand has its top bit set, where the kinds disagree: an
// arithmetic shift of 2**255 by 255 is all-ones, not 1, and a logical shift of
// -8 by 1 is a large positive value, not -4.
contract C {
    function f() public pure {
        uint a = 1 << 255;
        assert((a >> 255) == 1);
        int b = -8;
        assert((b >> 1) == -4);
    }
}
