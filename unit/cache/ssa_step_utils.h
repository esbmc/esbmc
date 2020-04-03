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

constant_int2tc create_signed_32_value_expr(unsigned value)
{
  signedbv_type2tc i32type(32);
  BigInt num(value);
  constant_int2tc expression(i32type, num);
  return expression;
}

add2tc create_unsigned_32_add_expr(expr2tc &side1, expr2tc &side2)
{
  unsignedbv_type2tc type(32);
  add2tc expression(type, side1, side2);
  return expression;
}

add2tc create_signed_32_add_expr(expr2tc &side1, expr2tc &side2)
{
  signedbv_type2tc type(32);
  add2tc expression(type, side1, side2);
  return expression;
}

equality2tc create_equality_relation(expr2tc &lhs, expr2tc &rhs)
{
  equality2tc relation(lhs, rhs);
  return relation;
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


// Pre-Built expressions
// These expressions will be used thorough the
// test cases as simple validations.
// The fuzzer should use the above method as a way to construct
// and validate expressions

namespace 
{
  symbol2tc Y = create_unsigned_32_symbol_expr("Y");
  symbol2tc X = create_unsigned_32_symbol_expr("X");
  constant_int2tc values[10];
  constant_int2tc neg_values[10];
  void init_values()
  {
    static bool is_initialized = false;
    if(is_initialized) return;
    for(int i = 0; i < 10; i++)
    {
      values[i] = create_signed_32_value_expr(i);
      neg_values[i] = create_signed_32_value_expr(0-i);
    }
    is_initialized = true;
  } 
}

// ((Y + y) + 7) == 9
inline expr2tc equality_1()
{
  add2tc add_1 = create_signed_32_add_expr(Y,X);
  add2tc add_2 = create_signed_32_add_expr(add_1,values[7]);
  equality2tc result = create_equality_relation(add_2, values[9]);
  return result;
}

// ((x + y) + 7) == 9
inline expr2tc equality_1_ordered()
{
  add2tc add_1 = create_signed_32_add_expr(X,Y);
  add2tc add_2 = create_signed_32_add_expr(add_1,values[7]);
  equality2tc result = create_equality_relation(add_2, values[9]);
  return result;
}

// ((x + y) + -2) == 0
inline expr2tc equality_1_green_normal()
{
  add2tc add_1 = create_signed_32_add_expr(X,Y);
  add2tc add_2 = create_signed_32_add_expr(add_1,neg_values[2]);
  equality2tc result = create_equality_relation(add_2, values[0]);
  return result;
}

// (1 + x) == 0
inline expr2tc equality_2()
{
  add2tc add_1 = create_signed_32_add_expr(values[1],X);
  equality2tc result = create_equality_relation(add_1, values[0]);
  return result;
}

// (x + 1) == 0
inline expr2tc equality_2_ordered()
{
  add2tc add_1 = create_signed_32_add_expr(X, values[1]);
  equality2tc result = create_equality_relation(add_1, values[0]);
  return result;
}

// (x + 1) == 0
inline expr2tc equality_2_green_normal()
{
  return equality_2_ordered();
}

// (y + 4) == 8
inline expr2tc equality_3()
{
  add2tc add_1 = create_signed_32_add_expr(Y,values[4]);
  equality2tc result = create_equality_relation(add_1, values[8]);
  return result;
}

// (y + 4) == 8
inline expr2tc equality_3_ordered()
{
  return equality_3();
}

// (y - 4) == 0
inline expr2tc equality_3_green_normal()
{
  add2tc add_1 = create_signed_32_add_expr(Y,neg_values[4]);
  equality2tc result = create_equality_relation(add_1, values[0]);
  return result;
}

// (x + 0) == 0
inline expr2tc equality_4()
{
  add2tc add_1 = create_signed_32_add_expr(X,values[0]);
  equality2tc result = create_equality_relation(add_1, values[0]);
  return result;
}

// (x + 0) == 0
inline expr2tc equality_4_ordered()
{
  return equality_4();
}

// (x + 0) == 0
inline expr2tc equality_4_green_normal()
{
  return equality_4();
}

#endif //ESBMC_SSA_STEP_UTILS_H
