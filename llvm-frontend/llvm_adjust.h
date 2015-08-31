/*
 * llvmadjust.h
 *
 *  Created on: Aug 30, 2015
 *      Author: mramalho
 */

#ifndef LLVM_FRONTEND_LLVM_ADJUST_H_
#define LLVM_FRONTEND_LLVM_ADJUST_H_

#include <context.h>
#include <std_expr.h>

class llvm_adjust
{
  public:
    llvm_adjust(contextt &_context);
    virtual ~llvm_adjust();

    bool adjust();

  private:
    contextt &context;

    void adjust_function(symbolt &symbol);

    void convert_exprt(exprt &expr);
    void convert_member(member_exprt &expr);
    void convert_expr_to_codet(exprt &expr);
};

#endif /* LLVM_FRONTEND_LLVM_ADJUST_H_ */
