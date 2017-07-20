/*
 * clang_c_convert.cpp
 *
 *  Created on: Jul 23, 2015
 *      Author: mramalho
 */

#include "clang_cpp_convert.h"

#include <clang/AST/DeclCXX.h>
#include <clang/AST/DeclFriend.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/StmtCXX.h>
#include <cpp/cpp_type2name.h>
#include <util/std_code.h>
#include <util/std_expr.h>

clang_cpp_convertert::clang_cpp_convertert(
  contextt &_context,
  std::vector<std::unique_ptr<clang::ASTUnit> > &_ASTs)
  : clang_c_convertert(_context, _ASTs)
{
}

bool clang_cpp_convertert::get_decl(
  const clang::Decl& decl,
  exprt &new_expr)
{
  new_expr = code_skipt();

  switch (decl.getKind())
  {
    case clang::Decl::LinkageSpec:
    {
      const clang::LinkageSpecDecl &lsd =
        static_cast<const clang::LinkageSpecDecl&>(decl);

      for(auto decl : lsd.decls())
        if(get_decl(*decl, new_expr))
          return true;
      break;
    }

    case clang::Decl::CXXRecord:
    {
      const clang::CXXRecordDecl &cxxrd =
        static_cast<const clang::CXXRecordDecl&>(decl);

      if(get_struct_union_class(cxxrd))
        return true;

      break;
    }

    default:
      return clang_c_convertert::get_decl(decl, new_expr);
  }

  return false;
}

bool clang_cpp_convertert::get_function(
  const clang::FunctionDecl &fd,
  exprt &new_expr)
{
  // Only convert instantiated functions/methods
  if(fd.isDependentContext())
    return false;

  return clang_c_convertert::get_function(fd, new_expr);
}

bool clang_cpp_convertert::get_struct_union_class(
  const clang::RecordDecl& rd)
{
  // Only convert instantiated functions/methods
  if(rd.isDependentContext())
    return false;

  return clang_c_convertert::get_struct_union_class(rd);
}

bool clang_cpp_convertert::get_struct_union_class_fields(
  const clang::RecordDecl &recordd,
  struct_union_typet &type)
{
  // If a struct is defined inside a extern C, it will be a RecordDecl
  const clang::CXXRecordDecl* cxxrd =
    llvm::dyn_cast<clang::CXXRecordDecl>(&recordd);
  if(cxxrd != nullptr)
  {
    // So this is a CXXRecordDecl, let's check for (virtual) base classes
    for(const auto &decl : cxxrd->bases())
    {
      // The base class is always a CXXRecordDecl
      const clang::CXXRecordDecl* base =
        decl.getType().getTypePtr()->getAsCXXRecordDecl();
      assert(base != nullptr);

      if(clang_c_convertert::get_struct_union_class_fields(*base, type))
        return true;
    }
  }

  return clang_c_convertert::get_struct_union_class_fields(recordd, type);
}

bool clang_cpp_convertert::get_struct_union_class_methods(
  const clang::RecordDecl &recordd,
  struct_union_typet &type)
{
  // If a struct is defined inside a extern C, it will be a RecordDecl
  const clang::CXXRecordDecl* cxxrd =
    llvm::dyn_cast<clang::CXXRecordDecl>(&recordd);
  if(cxxrd == nullptr)
    return false;

  for(const auto &decl : cxxrd->methods())
  {
    exprt dummy;
    if(get_decl(*decl, dummy))
      return true;
  }

  return false;
}
