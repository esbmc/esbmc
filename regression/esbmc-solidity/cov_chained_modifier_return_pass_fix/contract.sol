// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Pin: chained modifiers + value-returning function — `--branch-coverage-claims`
// must instrument and reach both modifier checks.
//
// Pre-fix bug: in `get_func_modifier`, when constructing the call to the
// next-modifier wrapper (`func_modifier`) at the end of each iteration,
// the call expression's type was left at `nil_typet`.  At the start of
// the next iteration's chained build, the substitution loop that
// rewrites `return <expr>` -> `aux_var = <expr>` checked
// `op->op0().type().id().as_string().empty()` to detect bare returns,
// and the un-typed call passed that check — the return-with-call got
// replaced with `code_skipt()`, dropping the inner-modifier call
// entirely.  Result: the inner modifier's body never executed under the
// outer modifier's harness, so its `_;`-following branches were
// reported PROVABLY unreachable by k-induction (false-completeness
// loss, downstream coverage gap).
//
// Fix: set `func_modifier.type() = aux_type.return_type()` BEFORE
// stuffing it into the return_expr (solidity_convert_modifier.cpp).
// Post-fix: both modA's `block.timestamp == 0` and modB's
// `msg.sender == address(0)` checks are reached (Branches:4 Reached:4).

contract C {
    uint256 public x;
    modifier modA() {
        if (block.timestamp == 0) revert();
        _;
    }
    modifier modB() {
        if (msg.sender == address(0)) revert();
        _;
    }
    function f() external modA modB returns (uint256) {
        x = 1;
        return 1;
    }
}
