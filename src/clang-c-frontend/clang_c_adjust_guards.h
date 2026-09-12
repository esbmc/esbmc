#ifndef ESBMC_CLANG_C_FRONTEND_CLANG_C_ADJUST_GUARDS_H
#define ESBMC_CLANG_C_FRONTEND_CLANG_C_ADJUST_GUARDS_H

#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>

/// The guards the IREP2 adjust arm tables dispatch on. Shared rather than
/// file-local because a second frontend's table names the same ones
/// (docs/roadmap/scope-clang-cpp-irep2.md §3.1); they read only the node.

/// The operators C admits over a complex operand: `mod` and the bitwise ones
/// are not among them, and `clang_c_adjust` aborts rather than lowering those.
inline bool is_binary_arith(const expr2tc &expr)
{
  return is_add2t(expr) || is_sub2t(expr) || is_mul2t(expr) || is_div2t(expr);
}

/// `-z` and GNU `~z` (conjugation) are the only unary operators clang leaves
/// carrying a complex type.
inline bool is_complex_unary(const expr2tc &expr)
{
  return (is_neg2t(expr) || is_bitnot2t(expr)) && is_complex_type(expr->type);
}

/// The operators clang_c_adjust routes through adjust_expr_binary_arithmetic.
inline bool is_arith_or_bitwise(const expr2tc &expr)
{
  return is_binary_arith(expr) || is_modulus2t(expr) || is_bitand2t(expr) ||
         is_bitor2t(expr) || is_bitxor2t(expr);
}

/// The shifts clang_c_adjust routes through adjust_expr_shifts. They are not
/// in is_arith_or_bitwise: C11 6.5.7p3 promotes each operand on its own and
/// takes the result type from the left, where the usual arithmetic conversions
/// would bring the two to a common type.
inline bool is_shift(const expr2tc &expr)
{
  return is_shl2t(expr) || is_ashr2t(expr) || is_lshr2t(expr);
}

/// The statements whose controlling expression clang_c_adjust converts to bool
/// (adjust_ifthenelse, adjust_while, adjust_for). `switch` is not among them:
/// its selector is an integer.
inline bool is_statement_with_condition(const expr2tc &expr)
{
  return is_code_ifthenelse2t(expr) || is_code_while2t(expr) ||
         is_code_dowhile2t(expr) || is_code_for2t(expr);
}

/// The comparisons clang_c_adjust routes through adjust_expr_rel. IREP2 already
/// types these bool, so only the operand half of that arm ports.
inline bool is_relational(const expr2tc &expr)
{
  return is_equality2t(expr) || is_notequal2t(expr) || is_lessthan2t(expr) ||
         is_lessthanequal2t(expr) || is_greaterthan2t(expr) ||
         is_greaterthanequal2t(expr);
}

/// The short-circuit operators, whose operands goto_convert's lowering rejects
/// unless they are boolean.
inline bool is_short_circuit(const expr2tc &expr)
{
  return is_and2t(expr) || is_or2t(expr) || is_not2t(expr);
}

/// Both spellings of a call: a bare `f(x);` statement is a sideeffect2t of kind
/// function_call rather than a code_function_call2t.
inline bool is_call_site(const expr2tc &expr)
{
  return is_code_function_call2t(expr) || is_sideeffect2t(expr);
}

/// The unary operators promote_unary_bool_operand claims: the complement of
/// is_complex_unary within the family. The chain spelled this exclusion as the
/// `else` of is_complex_unary, which is a strict subset of the same family;
/// stating it as a predicate holds in both the rewriting and the declining case
/// for the same reason, rather than relying on the first arm having mutated the
/// node out of the second's reach.
inline bool is_promotable_unary(const expr2tc &expr)
{
  return (is_neg2t(expr) || is_bitnot2t(expr)) && !is_complex_type(expr->type);
}

/// A cast the frontend marked as a derived->base conversion it could not route
/// through a "@base@" component, or the wrapper migrate_expr builds when the
/// marker sat on a node that is not a cast (#7025).
inline bool is_derived_to_base_cast(const expr2tc &expr)
{
  return is_typecast2t(expr) && !to_typecast2t(expr).derived_to_base.empty();
}

/// The mirror: a downcast whose operand points at a base subobject (#1866).
inline bool is_base_to_derived_cast(const expr2tc &expr)
{
  return is_typecast2t(expr) && to_typecast2t(expr).base_to_derived;
}

#endif
