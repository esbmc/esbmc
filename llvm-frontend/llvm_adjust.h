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

class llvm_adjust
{
  public:
    llvm_adjust(contextt &_context);
    virtual ~llvm_adjust();

    bool adjust();

  private:
    contextt &context;
    namespacet ns;

    void adjust_function(symbolt &symbol);

    void convert_exprt(exprt &expr);
    void convert_member(member_exprt &expr);
    void convert_pointer_arithmetic(exprt &expr);
    void convert_index(index_exprt &index);
    void convert_dereference(exprt &deref);
    void convert_expr_to_codet(exprt &expr);
    void convert_expr_function_identifier(exprt &expr);
    void convert_sizeof(exprt& expr);

    void make_index_type(exprt &expr);
};

#endif /* LLVM_FRONTEND_LLVM_ADJUST_H_ */
