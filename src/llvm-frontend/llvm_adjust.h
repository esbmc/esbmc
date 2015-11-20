/*
 * llvmadjust.h
 *
 *  Created on: Aug 30, 2015
 *      Author: mramalho
 */

#ifndef LLVM_FRONTEND_LLVM_ADJUST_H_
#define LLVM_FRONTEND_LLVM_ADJUST_H_

#include <context.h>
#include <namespace.h>
#include <std_expr.h>
#include <std_code.h>

class llvm_adjust
{
  public:
    llvm_adjust(contextt &_context)
      : context(_context),
        ns(namespacet(context))
    {
    }

    ~llvm_adjust()
    {
    }

    bool adjust();

  private:
    contextt &context;
    namespacet ns;

    void adjust_function(symbolt &symbol);
    void adjust_type(typet &type);

    void convert_builtin(symbolt& symbol);

    void convert_expr(exprt &expr);
    void convert_expr_main(exprt &expr);

    void convert_side_effect_assignment(exprt &expr);
    void convert_side_effect_function_call(
      side_effect_expr_function_callt &expr);
    void convert_side_effect_statement_expression(side_effect_exprt &expr);
    void convert_member(member_exprt &expr);
    void convert_expr_binary_arithmetic(exprt &expr);
    void convert_expr_unary_boolean(exprt &expr);
    void convert_expr_binary_boolean(exprt &expr);
    void convert_index(index_exprt &index);
    void convert_dereference(exprt &deref);
    void convert_address_of(exprt &expr);
    void convert_sizeof(exprt &expr);
    void convert_side_effect(side_effect_exprt &expr);
    void convert_symbol(exprt &expr);

    void convert_code(codet &code);
    void convert_expression(codet &code);
    void convert_label(code_labelt &code);
    void convert_block(codet &code);
    void convert_ifthenelse(codet &code);
    void convert_while(codet &code);
    void convert_for(codet &code);
    void convert_switch(codet &code);
    void convert_assign(codet &code);

    void make_index_type(exprt &expr);
};

#endif /* LLVM_FRONTEND_LLVM_ADJUST_H_ */
