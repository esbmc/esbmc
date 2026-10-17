// SPDX-License-Identifier: MIT
// Regression for the modifier `{ _; assert(...); }` `to_symbol_expr` crash.
// Pre-fix, the modifier-inline splice walker recursed into every operand
// including bare leaf symbols, and its first-line non-const
// `node.operands()` call lazy-allocated an empty operands sub-irep on
// the assert function-symbol (`c:@F@assert`). The poisoned symbol then
// failed the `id == symbol && !has_operands()` precondition of
// `to_symbol_expr` when `clang_c_adjust::do_special_functions` revisited
// it (clang_c_adjust_expr.cpp:892), aborting during Solidity conversion.
//
// `require(...)` and `revert()` were unaffected because they are
// intercepted at solidity_convert_expr.cpp:2161-2205 BEFORE the generic
// builtin path, so the splice walker never saw the bare stdlib symbol.
// `assert(...)` had no such intercept.
//
// Post-fix: a one-line guard at the splice walker's entry returns early
// for leaf nodes (`!node.has_operands()`), preventing lazy-allocation.
// The same fix simultaneously cleared pre-existing failing tests
// modifier_3 and modifier_4, which shared the identical root cause.
//
// Originally pinned as KNOWNBUG (commit 458b42beb0); flipped to CORE in
// the splice-walker guard fix commit.
pragma solidity >=0.8.0;

contract C {
    modifier g {
        _;
        assert(true);
    }

    function f() external g {}
}
