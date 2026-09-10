#include <clang-cpp-frontend/clang_cpp_adjust_irep2.h>
#include <clang-cpp-frontend/clang_cpp_code_gen.h>
#include <clang-cpp-frontend/clang_cpp_exception_id.h>

/// The guards live on the C pass's translation unit as file-local statics, so
/// they are re-declared here rather than shared: they read only the node, and a
/// second definition of a one-line predicate is cheaper than exporting them.
#include <clang-c-frontend/clang_c_adjust_guards.h>

/// A member whose type is code: a method named through `.` or `->`. A data
/// member is left alone, and so is a member with no component name.
static bool is_cpp_member_call(const expr2tc &expr)
{
  return is_member2t(expr) && is_code_type(expr->type) &&
         !to_member2t(expr).member.empty();
}

/// A source-level try/catch whose handler ids have not been computed yet. The
/// post-goto-convert CATCH marker carries no operands and is left alone.
static bool is_unresolved_cpp_catch(const expr2tc &expr)
{
  return is_code_cpp_catch2t(expr) &&
         to_code_cpp_catch2t(expr).operands.size() > 1 &&
         to_code_cpp_catch2t(expr).exception_list.empty();
}

static bool is_unresolved_cpp_throw(const expr2tc &expr)
{
  return is_code_cpp_throw2t(expr) &&
         to_code_cpp_throw2t(expr).exception_list.empty() &&
         !is_nil_expr(to_code_cpp_throw2t(expr).operand);
}

#define ARM(member)                                                            \
#  member,                                                                     \
    +[](clang_cpp_adjust_irep2 & self, expr2tc & expr) { self.member(expr); }

/// Inherited arms only, in the C pass's order. What C++ adds goes here as the
/// divergence census names it (scope-clang-cpp-irep2.md §3.1).
const clang_cpp_adjust_irep2::arm clang_cpp_adjust_irep2::arms[] = {
  {ARM(adjust_cpp_catch), is_unresolved_cpp_catch},
  {ARM(adjust_cpp_throw), is_unresolved_cpp_throw},
  {ARM(adjust_cpp_member), is_cpp_member_call},
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
  {ARM(adjust_derived_to_base), is_derived_to_base_cast},
  {ARM(adjust_base_to_derived), is_base_to_derived_cast},
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

void clang_cpp_adjust_irep2::adjust_cpp_member(expr2tc &expr)
{
  const member2t &m = to_member2t(expr);
  const symbolt *comp = ns.lookup(m.member);
  if (!comp)
  {
    // The legacy arm aborts here. Declining instead would hand goto_convert a
    // member callee it rejects, so the diagnostic is worth more than the
    // fall-through -- but it names the member, which the legacy message does.
    log_error(
      "adjust_cpp_member: unresolved C++ member component `{}` (source type "
      "id `{}`)",
      m.member,
      get_type_id(m.source_value->type));
    abort();
  }

  assert(comp->get_type().is_code());
  expr = symbol2tc(migrate_type(comp->get_type()), comp->id);
}

void clang_cpp_adjust_irep2::adjust_cpp_catch(expr2tc &expr)
{
  const code_cpp_catch2t &c = to_code_cpp_catch2t(expr);

  // One id per handler, parallel to operands[1..N]; the legacy arm keeps only
  // the leading id per handler and expands base classes at the throw site.
  std::vector<irep_idt> ids;
  for (std::size_t i = 1; i < c.operands.size(); i++)
  {
    std::vector<irep_idt> one;
    convert_exception_id(
      ns, migrate_type_back(c.operands[i]->type), "", one, true);
    ids.push_back(one.empty() ? irep_idt() : one.front());
  }

  expr = code_cpp_catch2tc(ids, c.operands, c.location);
}

void clang_cpp_adjust_irep2::adjust_cpp_throw(expr2tc &expr)
{
  const code_cpp_throw2t &th = to_code_cpp_throw2t(expr);

  // Every id the thrown type resolves to, most derived first, so a handler for
  // a base catches it.
  std::vector<irep_idt> ids;
  convert_exception_id(ns, migrate_type_back(th.operand->type), "", ids);

  expr = code_cpp_throw2tc(th.operand, ids, th.location);
}

void clang_cpp_adjust_irep2::gen_symbol_code(symbolt &symbol)
{
  // The legacy pass generates these *after* adjusting the body; here they are
  // generated before, so the assignments go through the arms like any other
  // statement rather than being migrated back out and in again.
  gen_vptr_initializations(context, symbol);
}
