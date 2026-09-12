// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// The converter emits >> without picking the shift kind, and IREP2 has no
// kind-less shift node, so the seam resolves it by the left operand's signedness.
contract C {
    function f() public pure {
        uint a = 8;
        uint b = a >> 2;
        assert(b == 2);
    }
}
