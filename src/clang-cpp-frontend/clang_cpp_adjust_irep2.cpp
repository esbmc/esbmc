#include <clang-cpp-frontend/clang_cpp_adjust_irep2.h>

/// The guards live on the C pass's translation unit as file-local statics, so
/// they are re-declared here rather than shared: they read only the node, and a
/// second definition of a one-line predicate is cheaper than exporting them.
#include <clang-c-frontend/clang_c_adjust_guards.h>

#define ARM(member)                                                            \
#  member,                                                                     \
    +[](clang_cpp_adjust_irep2 & self, expr2tc & expr) { self.member(expr); }

/// Inherited arms only, in the C pass's order. What C++ adds goes here as the
/// divergence census names it (scope-clang-cpp-irep2.md §3.1).
const clang_cpp_adjust_irep2::arm clang_cpp_adjust_irep2::arms[] = {
  {ARM(adjust_function_designators), nullptr},
  {ARM(adjust_boolean_operands), is_short_circuit},
  {ARM(adjust_call_callee), is_call_site},
  {ARM(adjust_call_arguments), is_call_site},
  {ARM(adjust_if_expr), is_if2t},
  {ARM(adjust_complex_arith), is_binary_arith},
  {ARM(adjust_vector_float_arith), is_binary_arith},
  {ARM(adjust_statement_condition), is_statement_with_condition},
  {ARM(hoist_for_init), is_code_for2t},
  {ARM(adjust_expression_statement), is_code_expression2t},
  {ARM(adjust_comma_type), is_code_comma2t},
  {ARM(adjust_struct), is_constant_struct2t},
  {ARM(adjust_array_subtype), is_constant_array2t},
  {ARM(adjust_decl_init), is_code_decl2t},
  {ARM(adjust_dereference), is_dereference2t},
  {ARM(adjust_complex_unary), is_complex_unary},
  {ARM(promote_unary_bool_operand), is_promotable_unary},
  {ARM(adjust_relational), is_relational},
  {ARM(adjust_special_functions), is_sideeffect2t},
  {ARM(adjust_binary_arith_operands), is_arith_or_bitwise},
  {ARM(adjust_shift_operands), is_shift},
  {ARM(adjust_plain_assignment), is_sideeffect_assign2t},
  {ARM(adjust_compound_assignment), is_sideeffect_assign2t},
  {ARM(adjust_address_of), is_address_of2t},
};

#undef ARM

std::vector<clang_cpp_adjust_irep2::arm_info>
clang_cpp_adjust_irep2::arm_order()
{
  std::vector<arm_info> order;
  order.reserve(std::size(arms));
  for (const arm &a : arms)
    order.push_back({a.name, a.when});
  return order;
}

void clang_cpp_adjust_irep2::adjust_sole_arms(expr2tc &expr)
{
  run_adjust_arms(*this, arms, expr);
}
