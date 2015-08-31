/*
 * llvmadjust.h
 *
 *  Created on: Aug 30, 2015
 *      Author: mramalho
 */

#ifndef LLVM_FRONTEND_LLVM_ADJUST_H_
#define LLVM_FRONTEND_LLVM_ADJUST_H_

#include <context.h>

class llvm_adjust
{
  public:
    llvm_adjust(contextt &_context);
    virtual ~llvm_adjust();

    bool adjust();

  private:
    contextt &context;
};

#endif /* LLVM_FRONTEND_LLVM_ADJUST_H_ */
