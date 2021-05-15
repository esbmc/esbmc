// Rafael Sá Menezes - 04/2020

#include <cache/algorithms/expr_variable_reordering.h>
#include <cache/algorithms/algorithms_utils.h>
#include <array>

namespace
{
std::array<expr2t::expr_ids, 2> binary_operations = {
  expr2t::expr_ids::add_id,
  expr2t::expr_ids::mul_id};

std::array<expr2t::expr_ids, 6> relations = {
  expr2t::expr_ids::equality_id,
  expr2t::expr_ids::notequal_id,
  expr2t::expr_ids::lessthan_id,
  expr2t::expr_ids::greaterthan_id,
  expr2t::expr_ids::lessthanequal_id,
  expr2t::expr_ids::greaterthanequal_id};
} // namespace

void expr_variable_reordering::run()
{
  auto expr_type = get_expr_type(expr);
  switch(expr_type)
  {
  case PARSE_AS::BIN_OP:
  {
    this->run_on_binop(expr);
    break;
  }
  case PARSE_AS::RELATION:
  {
    this->run_on_relation(expr);
    break;
  }
  case PARSE_AS::NEG:
  {
    this->run_on_negation(expr);
    break;
  }
  default:;
  }
}
expr_variable_reordering::PARSE_AS
expr_variable_reordering::get_expr_type(expr2tc &expr)
{
  static std::map<expr2t::expr_ids, PARSE_AS> expr_to_function;
  static bool map_initialized = false;

  if(!map_initialized)
  {
    for(auto i : binary_operations)
    {
      expr_to_function[i] = PARSE_AS::BIN_OP;
    }

    for(auto i : relations)
    {
      expr_to_function[i] = PARSE_AS::RELATION;
    }

    expr_to_function[expr2t::expr_ids::constant_int_id] = PARSE_AS::CONSTANT;

    expr_to_function[expr2t::expr_ids::symbol_id] = PARSE_AS::SYMBOL;

    expr_to_function[expr2t::expr_ids::neg_id] = PARSE_AS::NEG;
    expr_to_function[expr2t::expr_ids::not_id] = PARSE_AS::NEG;

    map_initialized = true;
  }
  if(expr_to_function.find(expr->expr_id) == expr_to_function.end())
  {
    return PARSE_AS::SKIP;
  }
  return expr_to_function[expr->expr_id];
}

void expr_variable_reordering::run_on_binop(expr2tc &expr) noexcept
{
  std::shared_ptr<arith_2ops> arith;
  arith = std::dynamic_pointer_cast<arith_2ops>(expr);
  // If arith is false then the cast failed
  assert(arith);

  /** Reorder both sides recursively
   *
   * The idea of this algorithm is to parse all symbols and values of a
   * expression with the same operator and rewrite it.
   *
   * 1. Create a list of symbols and values
   * 2. Order it in crescent order
   * 3. Change all values
   */

  expr_variable_reordering side1_reordering(arith->side_1);
  expr_variable_reordering side2_reordering(arith->side_2);

  side1_reordering.run();
  side2_reordering.run();

  symbols_vec symbols; // annotates all inner symbols
  values_vec values;   // annotates all inner values

  // 1. Create a list of symbols and values
  this->transverse_read_binop(arith, symbols, values);

  // 2. Order it in crescent order
  sort(symbols.begin(), symbols.end(), [](const auto &lhs, const auto &rhs) {
    return lhs->get_symbol_name() > rhs->get_symbol_name();
  });

  // A constant propagation should have been executed prior to this
  assert(values.size() <= 1);

  // 3. Change all values
  this->transverse_replace_binop(arith, symbols, values);
}

void expr_variable_reordering::run_on_negation(expr2tc &expr) noexcept
{
  if(expr->expr_id == expr2t::expr_ids::neg_id)
  {
    std::shared_ptr<arith_1op> arith;
    arith = std::dynamic_pointer_cast<arith_1op>(expr);
    // If arith is false then the cast failed
    assert(arith);
    expr_variable_reordering inner(arith->value);
    inner.run();
  }
  else if(expr->expr_id == expr2t::expr_ids::not_id)
  {
    std::shared_ptr<bool_1op> arith;
    arith = std::dynamic_pointer_cast<bool_1op>(expr);
    // If arith is false then the cast failed
    assert(arith);
    expr_variable_reordering inner(arith->value);
    inner.run();
  }
}

void expr_variable_reordering::run_on_relation(expr2tc &expr) noexcept
{
  std::shared_ptr<relation_data> relation;
  relation = std::dynamic_pointer_cast<relation_data>(expr);

  // If relation is false then the cast failed
  assert(relation);

  expr_variable_reordering side1(relation->side_1);
  side1.run();

  expr_variable_reordering side2(relation->side_2);
  side2.run();
}

inline void expr_variable_reordering::transverse_read_binop(
  const std::shared_ptr<arith_2ops> op,
  expr_variable_reordering::symbols_vec &symbols,
  expr_variable_reordering::values_vec &values)
{
  transverse_binop(op, symbols, values, TRANSVERSE_MODE::READ);
}

inline void expr_variable_reordering::transverse_replace_binop(
  const std::shared_ptr<arith_2ops> op,
  expr_variable_reordering::symbols_vec &symbols,
  expr_variable_reordering::values_vec &values)
{
  transverse_binop(op, symbols, values, TRANSVERSE_MODE::REPLACE);
}

void expr_variable_reordering::parse_arith_side(
  const std::shared_ptr<arith_2ops> op,
  expr_variable_reordering::symbols_vec &symbols,
  expr_variable_reordering::values_vec &values,
  expr_variable_reordering::TRANSVERSE_MODE mode,
  bool is_lhs)
{
  bool side_is_same_operand = is_lhs ? op->side_1->expr_id == op->expr_id
                                     : op->side_2->expr_id == op->expr_id;
  // Check if LHS is the same binary operation of parent
  if(side_is_same_operand)
  {
    std::shared_ptr<arith_2ops> arith;
    arith =
      std::dynamic_pointer_cast<arith_2ops>(is_lhs ? op->side_1 : op->side_2);
    transverse_binop(arith, symbols, values, mode);
  }

  switch(mode)
  {
  case TRANSVERSE_MODE::READ:
  {
    add_value(op, is_lhs, symbols, values);
    break;
  }
  case TRANSVERSE_MODE::REPLACE:
  {
    replace_value(op, is_lhs, symbols, values);
    break;
  }
  }
}

void expr_variable_reordering::transverse_binop(
  const std::shared_ptr<arith_2ops> op,
  expr_variable_reordering::symbols_vec &symbols,
  expr_variable_reordering::values_vec &values,
  expr_variable_reordering::TRANSVERSE_MODE mode)
{
  parse_arith_side(op, symbols, values, mode, true);  // LHS
  parse_arith_side(op, symbols, values, mode, false); // RHS
}

void expr_variable_reordering::add_value(
  const std::shared_ptr<arith_2ops> op,
  bool is_lhs,
  expr_variable_reordering::symbols_vec &symbols,
  expr_variable_reordering::values_vec &values)
{
  auto side_expr = is_lhs ? op->side_1 : op->side_2;
  auto expr_type = get_expr_type(side_expr);
  switch(expr_type)
  {
  case PARSE_AS::SYMBOL:
  {
    symbol2tc symbol;
    symbol = side_expr;
    symbols.push_back(symbol);
    break;
  }
  case PARSE_AS::CONSTANT:
  {
    constant_int2tc value;
    value = side_expr;
    values.push_back(value);
    break;
  }
  default:; // Continue parsing without adding anything
  }
}

void expr_variable_reordering::replace_value(
  const std::shared_ptr<arith_2ops> op,
  bool is_lhs,
  symbols_vec &symbols,
  values_vec &values)
{
  auto side_expr = is_lhs ? op->side_1 : op->side_2;
  auto expr_type = get_expr_type(side_expr);
  bool should_change =
    expr_type == PARSE_AS::SYMBOL || expr_type == PARSE_AS::CONSTANT;

  if(should_change)
  {
    expr2tc to_add;
    if(!values.empty())
    {
      to_add = values.back();
      values.pop_back();
    }
    else
    {
      to_add = symbols.back();
      symbols.pop_back();
    }
    if(is_lhs)
      op->side_1 = to_add;
    else
      op->side_2 = to_add;
  }
}