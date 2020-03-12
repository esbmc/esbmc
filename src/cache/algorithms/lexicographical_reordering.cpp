//
// Created by rafaelsa on 10/03/2020.
//

#include <stack>
#include "lexicographical_reordering.h"

namespace
{
const std::array<expr2t::expr_ids, 4> arith_expr_names = {
  expr2t::expr_ids::add_id,
  expr2t::expr_ids::mul_id,
  expr2t::expr_ids::ieee_add_id,
  expr2t::expr_ids::ieee_mul_id};

inline const bool is_arith_expr(const expr2t &expr)
{
  return std::count(
           arith_expr_names.begin(), arith_expr_names.end(), expr.expr_id) > 0;
}
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

inline const bool is_value_expr(const expr2t &expr)
{
  return std::count(
           value_expr_names.begin(), value_expr_names.end(), expr.expr_id) > 0;
}

const std::array<expr2t::expr_ids, 1> symbol_expr_names = {
  expr2t::expr_ids::symbol_id};

inline const bool is_symbolic_expr(const expr2t &expr)
{
  return std::count(
           symbol_expr_names.begin(), symbol_expr_names.end(), expr.expr_id) >
         0;
}

const std::array<expr2t::expr_ids, 80> ignored_expr_names = {
  // Binary operations that cannot be simple swapped
  expr2t::expr_ids::div_id,
  expr2t::expr_ids::ieee_div_id,
  expr2t::expr_ids::sub_id,
  expr2t::expr_ids::ieee_sub_id,

  // Other expr
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

typedef std::vector<symbol2tc> symbols_vec;
typedef std::vector<constant_int2tc> values_vec;

void replace_value(
  const std::shared_ptr<arith_2ops> op,
  bool is_side_1,
  symbols_vec &symbols,
  values_vec &values)
{
  auto side_to_check = is_side_1 ? op->side_1 : op->side_2;
  bool is_to_change =
    is_symbolic_expr(*side_to_check) || is_value_expr(*side_to_check);
  if(is_to_change)
  {
    expr2tc to_add;
    if(!symbols.empty())
    {
      to_add = symbols.back();
      symbols.pop_back();
    }
    else
    {
      to_add = values.back();
      values.pop_back();
    }
    if(is_side_1)
      op->side_1 = to_add;
    else
      op->side_2 = to_add;
  }
}

void add_value(
  const std::shared_ptr<arith_2ops> op,
  bool is_side_1,
  symbols_vec &symbols,
  values_vec &values)
{
  auto side_to_add = is_side_1 ? op->side_1 : op->side_2;
  if(is_symbolic_expr(*side_to_add))

  {
    symbol2tc symbol;
    symbol = side_to_add;
    symbols.push_back(symbol);
  }
  else if(is_value_expr(*side_to_add))
  {
    constant_int2tc value;
    value = side_to_add;
    values.push_back(value);
  }
}

void transverse_replacing_arith_op(
  const std::shared_ptr<arith_2ops> op,
  symbols_vec &symbols,
  values_vec &values)
{
  if(op->side_1->expr_id == op->expr_id)
  {
    std::shared_ptr<arith_2ops> arith;
    arith = std::dynamic_pointer_cast<arith_2ops>(op->side_1);
    transverse_replacing_arith_op(arith, symbols, values);
  }

  replace_value(op, true, symbols, values);
  if(op->side_2->expr_id == op->expr_id)
  {
    std::shared_ptr<arith_2ops> arith;
    arith = std::dynamic_pointer_cast<arith_2ops>(op->side_2);
    transverse_replacing_arith_op(arith, symbols, values);
  }
  replace_value(op, false, symbols, values);
}

void transverse_arith_op(
  const std::shared_ptr<arith_2ops> op,
  symbols_vec &symbols,
  values_vec &values)
{
  if(op->side_1->expr_id == op->expr_id)
  {
    std::shared_ptr<arith_2ops> arith;
    arith = std::dynamic_pointer_cast<arith_2ops>(op->side_1);
    transverse_arith_op(arith, symbols, values);
  }
  add_value(op, true, symbols, values);

  if(op->side_2->expr_id == op->expr_id)
  {
    std::shared_ptr<arith_2ops> arith;
    arith = std::dynamic_pointer_cast<arith_2ops>(op->side_2);
    transverse_arith_op(arith, symbols, values);
  }
  add_value(op, false, symbols, values);
}

} // namespace

bool lexicographical_reordering::should_swap(expr2tc &side1, expr2tc &side2)
{
  bool result = is_value_expr(*side1) && is_symbolic_expr(*side2);
  if(is_symbolic_expr(*side1) && is_symbolic_expr(*side2))
  {
    std::shared_ptr<symbol_data> symbol1;
    symbol1 = std::dynamic_pointer_cast<symbol_data>(side1);

    std::shared_ptr<symbol_data> symbol2;
    symbol2 = std::dynamic_pointer_cast<symbol_data>(side2);

    result = symbol1->get_symbol_name() > symbol2->get_symbol_name();
  }

  return result;
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

void lexicographical_reordering::run_on_assume(
  symex_target_equationt::SSA_stept &step)
{
  expr2tc &cond = step.cond;
  process_expr(cond);
}

void lexicographical_reordering::run_on_assignment(
  symex_target_equationt::SSA_stept &step)
{
  expr2tc &rhs = step.rhs;
  process_expr(rhs);
}

void lexicographical_reordering::run_on_assert(
  symex_target_equationt::SSA_stept &step)
{
  expr2tc &cond = step.cond;

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
  /* The reorder will check if LHS and RHS are symbols/values
   * the precedence will be x op y op z op value
   */
  std::shared_ptr<arith_2ops> arith;
  arith = std::dynamic_pointer_cast<arith_2ops>(expr);
  process_expr(arith->side_1);
  process_expr(arith->side_2);

  /**
   * The check are gonna be:
   *
   * 1. If the expression is BinOp X Y
   * 2. If The expression is BinOp BinOp X Y Z
   * 3. Check if constant propagation did not happen
   */

  symbols_vec symbols;
  values_vec values;
  transverse_arith_op(arith, symbols, values);
  sort(symbols.begin(), symbols.end(), [](const auto &lhs, const auto &rhs) {
    return lhs->get_symbol_name() > rhs->get_symbol_name();
  });

  if(values.size() > 1)
    throw std::runtime_error("Reordering expects max of one number");

  transverse_replacing_arith_op(arith, symbols, values);
}
