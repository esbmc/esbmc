/*
 * This file contains functions to resolve dynamic binding
 * when a clang::MemberExpr refers to a virtual/overriding method, e.g.:
 *  x->F
 *  where F represents a virtual/overriding method
 *
 *  clang::MemberExpr may represent x->F or x.F, but we don't care about dot operator.
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

bool clang_cpp_convertert::perform_virtual_dispatch(
  const clang::MemberExpr &member)
{
  // x.F() can't be a virtual dispatch
  if(!member.isArrow())
    return false;

  const clang::Decl &decl = *member.getMemberDecl();
  switch(decl.getKind())
  {
  // TODO: dtor might be virtual too
  //case clang::Decl::CXXDestructor:
  case clang::Decl::CXXMethod:
  {
    const clang::CXXMethodDecl &cxxmd =
      static_cast<const clang::CXXMethodDecl &>(decl);

    clang::LangOptions langOpts;
    // set LangOptions in C++ mode with RIIT enabled
    langOpts.CPlusPlus = 1;
    langOpts.RTTI = 1;
    if(
      member.performsVirtualDispatch(langOpts) &&
      cxxmd.getKind() != clang::Decl::CXXConstructor &&
      (is_md_virtual_or_overriding(cxxmd)))
      return true;

    break;
  }
  default:
    return false;
  }

  return false;
}

bool clang_cpp_convertert::is_md_virtual_or_overriding(
  const clang::CXXMethodDecl &cxxmd)
{
  return (
    cxxmd.isVirtual() ||
    cxxmd.begin_overridden_methods() != cxxmd.end_overridden_methods());
}

bool clang_cpp_convertert::is_fd_virtual_or_overriding(
  const clang::FunctionDecl &fd)
{
  if(const auto *md = llvm::dyn_cast<clang::CXXMethodDecl>(&fd))
    return is_md_virtual_or_overriding(*md);
  else
    return false;
}

bool clang_cpp_convertert::get_vft_binding_expr(
  const clang::MemberExpr &member,
  exprt &new_expr)
{
  /*
   * To turn `x->F` into `x->X@vtable_pointer->F`, we would need
   * a bunch of dereference expressions nested together.
   */
  // Let's start with base `x` dereferencing
  exprt base_deref;
  if(get_vft_binding_expr_base(member, base_deref))
    return true;

  // Then deal with X@vtable_pointer dereferencing
  exprt vtable_ptr_deref;
  get_vft_binding_expr_vtable_ptr(member, vtable_ptr_deref, base_deref);

  // Then deal with F dereferencing
  exprt f_expr;
  if(get_vft_binding_expr_function(member, f_expr, vtable_ptr_deref))
    return true;

  new_expr.swap(f_expr);
  return false;
}

bool clang_cpp_convertert::get_vft_binding_expr_base(
  const clang::MemberExpr &member,
  exprt &new_expr)
{
  exprt base;
  if(get_expr(*member.getBase(), base))
    return true;

  new_expr = dereference_exprt(base, base.type());
  new_expr.set("#lvalue", true);

  return false;
}

void clang_cpp_convertert::get_vft_binding_expr_vtable_ptr(
  const clang::MemberExpr &member,
  exprt &new_expr,
  const exprt &base_deref)
{
  // get the parent class id of the method to which this MemberExpr refers
  const auto md = llvm::dyn_cast<clang::CXXMethodDecl>(member.getMemberDecl());
  assert(md);
  std::string base_class_id = tag_prefix + md->getParent()->getNameAsString();

  std::string vtable_type_symb_id = vtable_type_prefix + base_class_id;
  typet vtable_type = symbol_typet(vtable_type_symb_id);

  std::string vtable_ptr_name = base_class_id + "::" + vtable_ptr_suffix;
  pointer_typet member_type(vtable_type);
  member_exprt deref_member(base_deref, vtable_ptr_name, member_type);
  deref_member.set("#lvalue", true);

  // we've got the deref type and member. Now we are ready to make the deref new_expr
  new_expr = dereference_exprt(deref_member, member_type);
  new_expr.set("#lvalue", true);
}

bool clang_cpp_convertert::get_vft_binding_expr_function(
  const clang::MemberExpr &member,
  exprt &new_expr,
  const exprt &vtable_ptr_deref)
{
  // get the parent class id of the method to which this MemberExpr refers
  const auto md = llvm::dyn_cast<clang::CXXMethodDecl>(member.getMemberDecl());
  assert(md);
  std::string base_class_id = tag_prefix + md->getParent()->getNameAsString();

  exprt comp;
  if(get_decl(*member.getMemberDecl(), comp))
    return true;

  /*
   * This is the component name as in vtable's type symbol
   * e.g. virtual_table::tag.Bird::c:@S@Bird@F@do_something#
   */
  std::string member_comp_name =
    vtable_type_prefix + base_class_id + "::" + comp.name().as_string();
  pointer_typet member_type(comp.type());
  member_exprt deref_member(vtable_ptr_deref, member_comp_name, member_type);
  deref_member.set("#lvalue", true);

  new_expr = dereference_exprt(deref_member, member_type);
  new_expr.set("#lvalue", true);

  return false;
}
