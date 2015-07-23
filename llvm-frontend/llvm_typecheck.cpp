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
  if(convert_top_level_decl())
    return true;

  return false;
}

bool llvm_typecheckt::convert_top_level_decl()
{
  // Iterate through each translation unit and their global symbols, creating
  // symbols as we go.

  for (auto &translation_unit : ASTs) {
    clang::ASTUnit::top_level_iterator it;
    for (it = translation_unit->top_level_begin();
        it != translation_unit->top_level_end(); it++) {
      switch ((*it)->getKind()) {
        case clang::Decl::Typedef:
        case clang::Decl::Function:
        case clang::Decl::Record:
        case clang::Decl::Var:
        default:
          std::cerr << "Unrecognized / unimplemented decl type ";
          std::cerr << (*it)->getDeclKindName() << std::endl;
          abort();
      }
    }
  }
}
