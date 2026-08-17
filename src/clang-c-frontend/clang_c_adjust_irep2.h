#pragma once

#include <util/symtab/context.h>
#include <util/symtab/namespace.h>
#include <irep2/irep2.h>

/// Phase 6 (C.3) IREP2-native adjuster for the C frontend.
///
/// The eventual replacement for `clang_c_adjust`, following the shape
/// `python_adjust` established (docs/roadmap/scope-clang-c-irep2.md §6): an
/// in-place recursive walk over `expr2tc` rather than the converter's
/// out-parameter seam.
///
/// At this stage the walk is deliberately **read-only**: it reads each code
/// symbol's IREP2 value and recurses, and never writes one back. That keeps the
/// pass inert by construction rather than by argument -- there is no write path
/// to be wrong -- while still exercising `migrate_expr` over every construct
/// the C corpus contains, since `symbolt::get_value2()` migrates the legacy
/// value on demand. A construct that cannot migrate aborts here instead of much
/// later.
///
/// Read-only also side-steps the round-trip losses `python_adjust` documents
/// (a bitfield's `#bitfield` flag, an explicit alignment attribute): those only
/// matter to a write-back, and C headers are exactly the place they occur.
///
/// Known limitation: the walk aborts on a union constant whose type is still a
/// by-name tag -- `migrate_expr` hands `migrate_type`'s `symbol_type2t` to
/// `constant_union2tc`, whose `is_union_type` assertion then fires. Running
/// after `clang_c_adjust` was meant to guarantee resolved aggregate types, as
/// it does for Python after `clang_cpp_adjust`; for C it does not. 12 of the
/// 1686 tests in regression/esbmc reach it. Left unguarded on purpose: this is
/// the defect the walk exists to surface, and the flag is opt-in.
class clang_c_adjust_irep2
{
public:
  /// @param sole_adjuster true under --clang-c-irep2-adjust-only, where this
  /// pass *replaces* clang_c_adjust. Work that substitutes for the legacy pass
  /// -- declaring an implicitly-declared callee (§70) -- must run only then:
  /// in shadow mode the legacy pass has already done it, and doing it twice
  /// adds conflicting symbols for library functions.
  explicit clang_c_adjust_irep2(contextt &_context, bool sole_adjuster = false)
    : context(_context), sole_adjuster(sole_adjuster)
  {
  }

  /// Walk every code symbol's IREP2 value. Returns false; there is no failure
  /// mode yet, and the signature matches `clang_c_adjust::adjust()` so the
  /// driver can call either.
  bool adjust();

  void adjust_expr(expr2tc &expr);

private:
  /// IREP2 form of clang_c_adjust::adjust_index's rewrite. The legacy arm keeps
  /// the operand recursion and returns before this point when the flag is on
  /// (scope-clang-c-irep2.md §19.2).
  void adjust_index(expr2tc &expr);

  /// IREP2 form of clang_c_adjust::adjust_member's rewrite: reach through a
  /// pointer base with a dereference, or an array base with a zero index.
  void adjust_member(expr2tc &expr);

  /// A call to a function with no visible declaration reaches here with no
  /// symbol in the context; clang_c_adjust::adjust_side_effect_function_call
  /// declares one. That is a symbol-table side effect rather than an expression
  /// rewrite, so it ports independently of the rest of that arm (§70).
  void declare_implicit_callee(const expr2tc &expr);

  /// IREP2 form of clang_c_adjust::adjust_expr_{unary,binary}_boolean's live
  /// half. Their type write is dead for C (§58.1); the operand conversion is
  /// not -- goto_convert's short-circuit lowering rejects a non-boolean operand
  /// outright. Portable here, unlike §58's dispatch-point shape, because this
  /// walk migrates once and does not round-trip per node (§78).
  void adjust_boolean_operands(expr2tc &expr);

  /// IREP2 form of the "do implicit dereference" step of
  /// clang_c_adjust::adjust_side_effect_function_call: goto_convert's
  /// do_function_call accepts a symbol or a dereference, so a pointer-typed
  /// callee -- a function-pointer struct member, say -- must be dereferenced
  /// first (§82).
  void adjust_call_callee(expr2tc &expr);

  /// IREP2 form of clang_c_adjust::adjust_expr_binary_arithmetic's complex
  /// branch: decompose `a op b` over a complex operand into per-component
  /// arithmetic and rebuild a (real, imag) pair. Unported, a complex `/`
  /// reaches goto_check's div_by_zero_check, which asks gen_zero for a complex
  /// zero and aborts -- the default path never gets there because it lowers to
  /// ieee_div, which is exempt from that check (§88).
  void adjust_complex_arith(expr2tc &expr);

  /// IREP2 form of clang_c_adjust::adjust_expr_unary_complex: `-z` negates both
  /// components, GNU `~z` conjugates. Unported, either reaches the solver still
  /// carrying a complex type and segfaults it (§90).
  void adjust_complex_unary(expr2tc &expr);

  /// IREP2 form of clang_c_adjust::adjust_if: a ternary's condition must be
  /// boolean before goto_convert lowers it, and its arms must agree with the
  /// node's type. `if2t` is the only value-level kind carrying a location
  /// (§49.2), so the rebuild passes the original through (§84).
  void adjust_if_expr(expr2tc &expr);

  /// IREP2 form of the `__builtin_`-prefixed half of
  /// clang_c_adjust::do_special_functions: fold a recognised builtin call to
  /// the node it denotes. These spellings are reserved, so unlike the
  /// name-matched family (is_name_matched_builtin) a program cannot supply its
  /// own definition and no shadows_user_definition query is needed (§90).
  void adjust_special_functions(expr2tc &expr);

  /// Arms that run only when this pass is the sole adjuster.
  void adjust_sole_arms(expr2tc &expr);

  contextt &context;
  const bool sole_adjuster;
  namespacet ns{context};
};

/// Shape-2 probe (scope-clang-c-irep2.md §57): run the IREP2 comma rewrite at
/// the point the legacy pass dispatches it, rather than in a second whole-
/// program pass. §55.4 attributes the comma arm's divergences to that ordering
/// alone; this is the experiment that decides it.
void adjust_comma_at_dispatch(exprt &expr, const namespacet &ns);
