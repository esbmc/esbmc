/*
 * clang_c_adjust.h
 *
 *  Created on: Aug 30, 2015
 *      Author: mramalho
 */


#ifndef CLANG_C_FRONTEND_CLANG_C_ADJUST_H_
#define CLANG_C_FRONTEND_CLANG_C_ADJUST_H_

#include <context.h>
#include <namespace.h>
#include <std_expr.h>
#include <std_code.h>

class clang_c_adjust
{
  public:
    clang_c_adjust(contextt &_context);
    virtual ~clang_c_adjust() = default;

    bool adjust();

  private:
    contextt &context;
    namespacet ns;

    void adjust_symbol(symbolt &symbol);
    void adjust_type(typet &type);

    void adjust_expr(exprt &expr);

    void adjust_side_effect_assignment(exprt &expr);
    void adjust_side_effect_function_call(
      side_effect_expr_function_callt &expr);
    void adjust_side_effect_statement_expression(side_effect_exprt &expr);
    void adjust_member(member_exprt &expr);
    void adjust_expr_binary_arithmetic(exprt &expr);
    void adjust_expr_unary_boolean(exprt &expr);
    void adjust_expr_binary_boolean(exprt &expr);
    void adjust_expr_rel(exprt &expr);
    void adjust_float_arith(exprt &expr);
    void adjust_index(index_exprt &index);
    void adjust_dereference(exprt &deref);
    void adjust_address_of(exprt &expr);
    void adjust_sizeof(exprt &expr);
    void adjust_side_effect(side_effect_exprt &expr);
    void adjust_symbol(exprt &expr);
    void adjust_comma(exprt &expr);
    void adjust_builtin_va_arg(exprt &expr);

    void adjust_function_call_arguments(
      side_effect_expr_function_callt &expr);

    void adjust_code(codet &code);
    void adjust_ifthenelse(codet &code);
    void adjust_while(codet &code);
    void adjust_for(codet &code);
    void adjust_switch(codet &code);
    void adjust_assign(codet &code);
    void adjust_decl(codet &code);

    void adjust_operands(exprt &expr);

    void adjust_argc_argv(const symbolt &main_symbol);

    void make_index_type(exprt &expr);
    void do_special_functions(side_effect_expr_function_callt &expr);
};

#endif /* CLANG_C_FRONTEND_CLANG_C_ADJUST_H_ */
