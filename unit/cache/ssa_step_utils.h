//
// Created by rafaelsa on 12/03/2020.
//

#ifndef ESBMC_SSA_STEP_UTILS_H
#define ESBMC_SSA_STEP_UTILS_H


#include <cache/ssa_step_algorithm.h>

// TODO: Test all functions from this file

bool is_symbols_equal(expr2tc &v1, expr2tc &v2)
{
  assert(v1->expr_id == expr2t::expr_ids::symbol_id);
  assert(v2->expr_id == expr2t::expr_ids::symbol_id);

  std::shared_ptr<symbol_data> symbol1;
  symbol1 = std::dynamic_pointer_cast<symbol_data>(v1);

  std::shared_ptr<symbol_data> symbol2;
  symbol2 = std::dynamic_pointer_cast<symbol_data>(v2);

  return symbol1->get_symbol_name() == symbol2->get_symbol_name();
}

bool is_unsigned_equal(expr2tc &v1, expr2tc &v2)
{
  assert(v1->expr_id == expr2t::expr_ids::constant_int_id);
  assert(v2->expr_id == expr2t::expr_ids::constant_int_id);

  constant_int2tc value1 = v1;
  constant_int2tc value2 = v2;

  return value1->value == value2->value;
}

symbol2tc create_unsigned_32_symbol_expr(std::string name)
{
  unsignedbv_type2tc u32type(32);
  irep_idt var(name);
  symbol2tc expression(u32type, var);
  return expression;
}

constant_int2tc create_unsigned_32_value_expr(unsigned value)
{
  unsignedbv_type2tc u32type(32);
  BigInt num(value);
  constant_int2tc expression(u32type, num);
  return expression;
}

add2tc create_unsigned_32_add_expr(expr2tc &side1, expr2tc &side2)
{
  unsignedbv_type2tc type(32);
  add2tc expression(type, side1, side2);
  return expression;
}

void create_assignment(symex_target_equationt::SSA_stepst &output, expr2tc &rhs)
{
  symex_target_equationt::SSA_stept step1;
  step1.type = goto_trace_stept::ASSIGNMENT;
  step1.rhs = rhs;
  output.push_back(step1);
}

void create_assumption(symex_target_equationt::SSA_stepst &output, expr2tc &cond)
{
  symex_target_equationt::SSA_stept step1;
  step1.type = goto_trace_stept::ASSUME;
  step1.cond = cond;
  output.push_back(step1);
}

void create_assert(symex_target_equationt::SSA_stepst &output, expr2tc &cond)
{
  symex_target_equationt::SSA_stept step1;
  step1.type = goto_trace_stept::ASSERT;
  step1.cond = cond;
  output.push_back(step1);
}

#endif //ESBMC_SSA_STEP_UTILS_H
