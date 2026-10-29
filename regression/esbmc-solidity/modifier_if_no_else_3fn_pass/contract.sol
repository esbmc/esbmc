// SPDX-License-Identifier: MIT
// Regression for the no-else modifier `_;` empty-body GOTO crash. Pre-fix,
// solidity_convert_modifier.cpp's splice_placeholders walker erased the
// single `_;` operand at the parent ifthenelse's then-slot and inserted
// N=0 operands from the empty wrapper body — collapsing the 2-op
// `ifthenelse(cond, _;)` to a 1-op shape that goto_convert.cpp:1647
// rejects. Post-fix, the splice wraps body_exprt's operands in a single
// code_blockt when the parent is fixed-arity, preserving arity even at
// N=0 so the lowered IR is `if (cond) {}`.
//
// Originally pinned as KNOWNBUG (commit 43840737e0); flipped to CORE in
// the splice fix commit. The "3+ functions" framing in the original
// report was observational — the crash actually fires on f1's modifier
// expansion before f2/f3 are processed.
pragma solidity >=0.8.0;

contract C {
    mapping(address => bool) ok;

    modifier g {
        if (ok[msg.sender]) _;
    }

    function f1() external g {}
    function f2() external g {}
    function f3() external g {}
}
