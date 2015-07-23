/*
 * llvmtypecheck.h
 *
 *  Created on: Jul 23, 2015
 *      Author: mramalho
 */

#ifndef LLVM_FRONTEND_LLVM_TYPECHECK_H_
#define LLVM_FRONTEND_LLVM_TYPECHECK_H_

#include <context.h>

#include <clang/Frontend/ASTUnit.h>

class llvm_typecheckt
{
public:
  llvm_typecheckt(contextt &_context);

  virtual ~llvm_typecheckt();

  bool typecheck();

  std::vector<std::unique_ptr<clang::ASTUnit> > ASTs;

private:
  contextt &context;
};

#endif /* LLVM_FRONTEND_LLVM_TYPECHECK_H_ */
