/*
 * llvmtypecheck.cpp
 *
 *  Created on: Jul 23, 2015
 *      Author: mramalho
 */

#include "llvm_typecheck.h"

llvm_typecheckt::llvm_typecheckt(contextt &_context)
  : context(_context)
{
}

llvm_typecheckt::~llvm_typecheckt()
{
}

bool llvm_typecheckt::typecheck()
{
  return true;
}
