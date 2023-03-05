/*
 * This file contains functions to resolve dynamic binding
 * when a clang::MemberExpr refers to a virtual/overriding method, e.g.:
 *  x->F or X.F.
 *  where F represents a virtual/overriding method
 */

#include <util/compiler_defs.h>
// Remove warnings from Clang headers
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <clang/AST/Attr.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/DeclFriend.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/QualTypeNames.h>
#include <clang/AST/Type.h>
#include <clang/Index/USRGeneration.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/AST/ParentMapContext.h>
#include <llvm/Support/raw_os_ostream.h>
CC_DIAGNOSTIC_POP()

#include <clang-cpp-frontend/clang_cpp_convert.h>
#include <util/expr_util.h>

bool clang_cpp_convertert::check_member_expr_virtual_overriding(
  const clang::Decl &decl)
{
  switch(decl.getKind())
  {
  // TODO: dtor might be virtual too
  //case clang::Decl::CXXDestructor:
  case clang::Decl::CXXMethod:
  {
    break;
  }
  default:
    return false;
  }

  const clang::CXXMethodDecl &cxxmd =
    static_cast<const clang::CXXMethodDecl &>(decl);

  // TODO: might be a good idea to use `performsVirtualDispatch(const LangOptions &LO)`?
  // These conditions indicate a member function call referring to a virtual/overriding method
  if(
    cxxmd.isVirtual() ||
    cxxmd.begin_overridden_methods() != cxxmd.end_overridden_methods())
    return true;

  return false;
}
