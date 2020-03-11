//
// Created by rafaelsa on 10/03/2020.
//

#include "lexicographical_reordering.h"

namespace
{
const std::array<expr2t::expr_ids, 8> arith_expr_names = {
  expr2t::expr_ids::add_id,
  expr2t::expr_ids::sub_id,
  expr2t::expr_ids::mul_id,
  expr2t::expr_ids::div_id,
  expr2t::expr_ids::ieee_add_id,
  expr2t::expr_ids::ieee_sub_id,
  expr2t::expr_ids::ieee_div_id,
  expr2t::expr_ids::ieee_mul_id};

const std::array<expr2t::expr_ids, 6> relation_expr_names = {
  expr2t::expr_ids::equality_id,
  expr2t::expr_ids::notequal_id,
  expr2t::expr_ids::lessthan_id,
  expr2t::expr_ids::greaterthan_id,
  expr2t::expr_ids::lessthanequal_id,
  expr2t::expr_ids::greaterthanequal_id};

const std::array<expr2t::expr_ids, 9> value_expr_names = {
  expr2t::expr_ids::constant_int_id,
  expr2t::expr_ids::constant_fixedbv_id,
  expr2t::expr_ids::constant_floatbv_id,
  expr2t::expr_ids::constant_bool_id,
  expr2t::expr_ids::constant_string_id,
  expr2t::expr_ids::constant_struct_id,
  expr2t::expr_ids::constant_union_id,
  expr2t::expr_ids::constant_array_id,
  expr2t::expr_ids::constant_array_of_id};

const std::array<expr2t::expr_ids, 1> symbol_expr_names = {
  expr2t::expr_ids::symbol_id};

const std::array<expr2t::expr_ids, 76> ignored_expr_names = {
  expr2t::expr_ids::typecast_id,
  expr2t::expr_ids::bitcast_id,
  expr2t::expr_ids::nearbyint_id,
  expr2t::expr_ids::if_id,
  expr2t::expr_ids::not_id,
  expr2t::expr_ids::and_id,
  expr2t::expr_ids::or_id,
  expr2t::expr_ids::xor_id,
  expr2t::expr_ids::implies_id,
  expr2t::expr_ids::bitand_id,
  expr2t::expr_ids::bitor_id,
  expr2t::expr_ids::bitxor_id,
  expr2t::expr_ids::bitnand_id,
  expr2t::expr_ids::bitnor_id,
  expr2t::expr_ids::bitnxor_id,
  expr2t::expr_ids::bitnot_id,
  expr2t::expr_ids::lshr_id,
  expr2t::expr_ids::neg_id,
  expr2t::expr_ids::abs_id,
  expr2t::expr_ids::ieee_fma_id,
  expr2t::expr_ids::ieee_sqrt_id,
  expr2t::expr_ids::popcount_id,
  expr2t::expr_ids::bswap_id,
  expr2t::expr_ids::modulus_id,
  expr2t::expr_ids::shl_id,
  expr2t::expr_ids::ashr_id,
  expr2t::expr_ids::dynamic_object_id,
  expr2t::expr_ids::same_object_id,
  expr2t::expr_ids::pointer_offset_id,
  expr2t::expr_ids::pointer_object_id,
  expr2t::expr_ids::address_of_id,
  expr2t::expr_ids::byte_extract_id,
  expr2t::expr_ids::byte_update_id,
  expr2t::expr_ids::with_id,
  expr2t::expr_ids::member_id,
  expr2t::expr_ids::index_id,
  expr2t::expr_ids::isnan_id,
  expr2t::expr_ids::overflow_id,
  expr2t::expr_ids::overflow_cast_id,
  expr2t::expr_ids::overflow_neg_id,
  expr2t::expr_ids::unknown_id,
  expr2t::expr_ids::invalid_id,
  expr2t::expr_ids::null_object_id,
  expr2t::expr_ids::dereference_id,
  expr2t::expr_ids::valid_object_id,
  expr2t::expr_ids::deallocated_obj_id,
  expr2t::expr_ids::dynamic_size_id,
  expr2t::expr_ids::sideeffect_id,
  expr2t::expr_ids::code_block_id,
  expr2t::expr_ids::code_assign_id,
  expr2t::expr_ids::code_init_id,
  expr2t::expr_ids::code_decl_id,
  expr2t::expr_ids::code_dead_id,
  expr2t::expr_ids::code_printf_id,
  expr2t::expr_ids::code_expression_id,
  expr2t::expr_ids::code_return_id,
  expr2t::expr_ids::code_skip_id,
  expr2t::expr_ids::code_free_id,
  expr2t::expr_ids::code_goto_id,
  expr2t::expr_ids::object_descriptor_id,
  expr2t::expr_ids::code_function_call_id,
  expr2t::expr_ids::code_comma_id,
  expr2t::expr_ids::invalid_pointer_id,
  expr2t::expr_ids::code_asm_id,
  expr2t::expr_ids::code_cpp_del_array_id,
  expr2t::expr_ids::code_cpp_delete_id,
  expr2t::expr_ids::code_cpp_catch_id,
  expr2t::expr_ids::code_cpp_throw_id,
  expr2t::expr_ids::code_cpp_throw_decl_id,
  expr2t::expr_ids::code_cpp_throw_decl_end_id,
  expr2t::expr_ids::isinf_id,
  expr2t::expr_ids::isnormal_id,
  expr2t::expr_ids::isfinite_id,
  expr2t::expr_ids::signbit_id,
  expr2t::expr_ids::concat_id,
  expr2t::expr_ids::extract_id};
} // namespace

lexicographical_reordering::lexicographical_reordering(
  symex_target_equationt::SSA_stepst &steps)
  : ssa_step_algorithm(steps)
{
}

void lexicographical_reordering::process_expr(expr2tc &rhs)
{
  typedef std::function<void(expr2tc &)> expr_function;

  static std::map<expr2t::expr_ids, expr_function> expr_to_function;
  static bool map_initialized = false;

  if(!map_initialized)
  {
    const auto arith_func = [this](expr2tc &expr) { run_on_arith(expr); };

    const auto relation_func = [this](expr2tc &expr) { run_on_relation(expr); };

    const auto symbol_func = [this](expr2tc &expr) { run_on_symbol(expr); };

    const auto value_func = [this](expr2tc &expr) { run_on_value(expr); };

    const auto ignored_func = [this](expr2tc &expr) {};

    for(auto i : arith_expr_names)
    {
      expr_to_function[i] = arith_func;
    }

    for(auto i : relation_expr_names)
    {
      expr_to_function[i] = relation_func;
    }

    for(auto i : value_expr_names)
    {
      expr_to_function[i] = value_func;
    }

    for(auto i : symbol_expr_names)
    {
      expr_to_function[i] = symbol_func;
    }

    for(auto i : ignored_expr_names)
    {
      expr_to_function[i] = ignored_func;
    }
    map_initialized = true;
  }

  if(expr_to_function.find(rhs->expr_id) == expr_to_function.end())
  {
    std::string msg("Map does not contain %d", rhs->expr_id);
    throw std::range_error(msg);
  }
  expr_to_function[rhs->expr_id](rhs);
}

void lexicographical_reordering::run_on_assert(
  symex_target_equationt::SSA_stept &step)
{
  expr2tc &cond = step.cond;
  std::string &comment = step.comment;

  // First assert irep should begin with an implies
  assert(cond->expr_id == expr2t::expr_ids::implies_id);

  // LHS only holds the guard which is not useful. So we parse RHS
  std::shared_ptr<logic_2ops> implies;
  implies = std::dynamic_pointer_cast<logic_2ops>(cond);
  expr2tc &rhs(implies->side_2);
  process_expr(rhs);
}

void lexicographical_reordering::run_on_relation(expr2tc &expr)
{
  std::shared_ptr<relation_data> relation;
  relation = std::dynamic_pointer_cast<relation_data>(expr);
  process_expr(relation->side_1);
  process_expr(relation->side_2);
}
void lexicographical_reordering::run_on_arith(expr2tc &expr)
{
  std::shared_ptr<arith_2ops> arith;
  arith = std::dynamic_pointer_cast<arith_2ops>(expr);
  // HERE THE REORDER SHOULD TAKE PLACE
  std::cout << "BEFORE\n";
  arith->dump();
  auto ref = arith->side_2;
  arith->side_2 = arith->side_1;
  arith->side_1 = ref;
  std::cout << "AFTER\n";
  arith->dump();
  process_expr(arith->side_1);
  process_expr(arith->side_2);
}
void lexicographical_reordering::run_on_symbol(expr2tc &expr)
{
  std::shared_ptr<symbol_data> symbol;
  symbol = std::dynamic_pointer_cast<symbol_data>(expr);
}
void lexicographical_reordering::run_on_value(expr2tc &expr)
{
}
