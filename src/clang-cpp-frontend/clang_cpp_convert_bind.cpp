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
#include <util/expr_util.cpp>

bool clang_cpp_convertert::check_member_expr_virtual_overriding()
{
  // Function to check
  assert(!"Got MemberExpr to virtual/overriding method");
  return false;
}
