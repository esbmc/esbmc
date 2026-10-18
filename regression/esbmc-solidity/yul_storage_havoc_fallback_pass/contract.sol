// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    uint256 x;

    // Pins the all-or-nothing rule: a literal-slot sstore (no `.slot`
    // ext-ref to resolve) makes the precise lowerer abort and the
    // legacy havoc fallback fires with a `[approx]` warning. The
    // tautology `x == x` then survives any havoc.
    function check(uint256 v) public {
        assembly {
            sstore(0, v)
        }
        assert(x == x);
    }
}
