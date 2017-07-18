/*
 * clang_c_adjust.h
 *
 *  Created on: Aug 30, 2015
 *      Author: mramalho
 */


#ifndef CLANG_C_FRONTEND_CLANG_C_ADJUST_H_
#define CLANG_C_FRONTEND_CLANG_C_ADJUST_H_

#include <util/context.h>
#include <util/namespace.h>
#include <util/std_code.h>
#include <util/std_expr.h>

class clang_c_adjust
{
public:
  clang_c_adjust(contextt &_context);
  virtual ~clang_c_adjust() = default;

  bool adjust();

protected:
  contextt &context;
  namespacet ns;

  virtual void adjust_symbol(symbolt &symbol);
  virtual void adjust_type(typet &type);

  virtual void adjust_expr(exprt &expr);

  virtual void adjust_side_effect_assignment(exprt &expr);
  virtual void adjust_side_effect_function_call(
    side_effect_expr_function_callt &expr);
  virtual void adjust_side_effect_statement_expression(side_effect_exprt &expr);
  virtual void adjust_member(member_exprt &expr);
  virtual void adjust_expr_binary_arithmetic(exprt &expr);
  virtual void adjust_expr_unary_boolean(exprt &expr);
  virtual void adjust_expr_binary_boolean(exprt &expr);
  virtual void adjust_expr_rel(exprt &expr);
  virtual void adjust_float_arith(exprt &expr);
  virtual void adjust_index(index_exprt &index);
  virtual void adjust_dereference(exprt &deref);
  virtual void adjust_address_of(exprt &expr);
  virtual void adjust_sizeof(exprt &expr);
  virtual void adjust_side_effect(side_effect_exprt &expr);
  virtual void adjust_symbol(exprt &expr);
  virtual void adjust_comma(exprt &expr);
  virtual void adjust_builtin_va_arg(exprt &expr);

  virtual void adjust_function_call_arguments(
    side_effect_expr_function_callt &expr);

  virtual void adjust_code(codet &code);
  virtual void adjust_ifthenelse(codet &code);
  virtual void adjust_while(codet &code);
  virtual void adjust_for(codet &code);
  virtual void adjust_switch(codet &code);
  virtual void adjust_assign(codet &code);
  virtual void adjust_decl(codet &code);

  virtual void adjust_operands(exprt &expr);

  virtual void adjust_argc_argv(const symbolt &main_symbol);

  virtual void do_special_functions(side_effect_expr_function_callt &expr);
};

#endif /* CLANG_C_FRONTEND_CLANG_C_ADJUST_H_ */
