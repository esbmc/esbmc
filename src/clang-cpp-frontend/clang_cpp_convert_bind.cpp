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
  if (!member.isArrow())
    return false;

  const clang::Decl &decl = *member.getMemberDecl();
  switch (decl.getKind())
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
    if (
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
  if (const auto *md = llvm::dyn_cast<clang::CXXMethodDecl>(&fd))
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
  if (get_vft_binding_expr_base(member, base_deref))
    return true;

  // Then deal with X@vtable_pointer dereferencing
  exprt vtable_ptr_deref;
  get_vft_binding_expr_vtable_ptr(member, vtable_ptr_deref, base_deref);

  // Then deal with F dereferencing
  exprt f_expr;
  if (get_vft_binding_expr_function(member, f_expr, vtable_ptr_deref))
    return true;

  new_expr.swap(f_expr);
  return false;
}

bool clang_cpp_convertert::get_vft_binding_expr_base(
  const clang::MemberExpr &member,
  exprt &new_expr)
{
  exprt base;
  if (get_expr(*member.getBase(), base))
    return true;

  new_expr = dereference_exprt(base, base.type());

  return false;
}

// Build an lvalue member access for the vtable pointer named @p vptr_name on
// the object lvalue @p object. Under Itanium primary-base sharing the vptr may
// not be a direct member of @p object's class: it lives in a (possibly nested)
// base subobject. Navigate into the base-subobject chain until the vptr is
// found. Returns false on success.
bool clang_cpp_convertert::build_vptr_member_access(
  const exprt &object,
  const irep_idt &vptr_name,
  const typet &vptr_member_type,
  exprt &result)
{
  const typet &t = ns.follow(object.type());
  if (t.id() != "struct")
    return true;
  const struct_typet &st = to_struct_type(t);

  // Direct vptr member with the exact name?
  for (const auto &c : st.components())
    if (c.get_name() == vptr_name)
    {
      result = member_exprt(object, vptr_name, vptr_member_type);
      return false;
    }

  // Recurse into base subobjects (the shared vptr lives in the primary base).
  for (const auto &c : st.components())
  {
    if (!c.get_bool("#is_base_subobject"))
      continue;
    exprt sub = member_exprt(object, c.get_name(), c.type());
    if (!build_vptr_member_access(sub, vptr_name, vptr_member_type, result))
      return false;
  }

  // Under primary-base sharing the class has a single vtable pointer at offset
  // 0 carrying the primary base's name; a derived-class virtual call (whose
  // name differs) resolves to that same slot. Fall back to the first vptr
  // component. The caller supplies the read type (the most-derived merged
  // vtable type when one exists), so the by-name index resolves correctly.
  for (const auto &c : st.components())
    if (c.get_bool("is_vtptr"))
    {
      result = member_exprt(object, c.get_name(), vptr_member_type);
      return false;
    }
  return true;
}

void clang_cpp_convertert::get_vft_binding_expr_vtable_ptr(
  const clang::MemberExpr &member,
  exprt &new_expr,
  const exprt &base_deref)
{
  // get the parent class id of the method to which this MemberExpr refers
  const auto md = llvm::dyn_cast<clang::CXXMethodDecl>(member.getMemberDecl());
  assert(md);
  std::string base_class_id, base_class_name;
  get_decl_name(*md->getParent(), base_class_name, base_class_id);

  std::string vtable_type_symb_id = vtable_type_prefix + base_class_id;
  typet vtable_type = symbol_typet(vtable_type_symb_id);

  std::string vtable_ptr_name = base_class_id + "::" + vtable_ptr_suffix;
  pointer_typet member_type(vtable_type);

  // Under Itanium primary-base sharing the shared vtable pointer (offset 0)
  // points at the receiver's most-derived MERGED vtable, which holds every
  // reachable slot under its original fully-qualified name. When the static
  // receiver type has such a merged vtable, read the pointer as that merged
  // type so the by-name index (using the method's own fully-qualified slot
  // name) resolves both inherited and own methods through the one shared slot.
  pointer_typet read_type = member_type;
  {
    const typet &recv = ns.follow(base_deref.type());
    if (recv.id() == "struct")
    {
      const irep_idt merged_id = vtable_type_prefix +
                                 to_struct_type(recv).tag().as_string() +
                                 "::merged";
      if (context.find_symbol(merged_id) != nullptr)
        read_type = pointer_typet(symbol_typet(merged_id));
    }
  }

  // The vtable pointer may be a direct member or, under primary-base sharing,
  // nested inside a base subobject. Navigate to it.
  exprt deref_member;
  if (build_vptr_member_access(
        base_deref, vtable_ptr_name, read_type, deref_member))
    // Fall back to a direct access (preserves prior behaviour if not found).
    deref_member = member_exprt(base_deref, vtable_ptr_name, read_type);

  // we've got the deref type and member. Now we are ready to make the deref new_expr
  new_expr = dereference_exprt(deref_member, read_type);
}

bool clang_cpp_convertert::get_vft_binding_expr_function(
  const clang::MemberExpr &member,
  exprt &new_expr,
  const exprt &vtable_ptr_deref)
{
  // get the parent class id of the method to which this MemberExpr refers
  const auto md = llvm::dyn_cast<clang::CXXMethodDecl>(member.getMemberDecl());
  assert(md);
  std::string base_class_id, base_class_name;
  get_decl_name(*md->getParent(), base_class_name, base_class_id);

  exprt comp;
  if (get_decl(*member.getMemberDecl(), comp))
    return true;

  /*
   * This is the component name as in vtable's type symbol
   * e.g. virtual_table::tag.Bird::c:@S@Bird@F@do_something#
   */
  std::string member_comp_name =
    vtable_type_prefix + base_class_id + "::" + comp.name().as_string();
  pointer_typet member_type(comp.type());
  member_exprt deref_member(vtable_ptr_deref, member_comp_name, member_type);

  new_expr = dereference_exprt(deref_member, member_type);

  return false;
}
