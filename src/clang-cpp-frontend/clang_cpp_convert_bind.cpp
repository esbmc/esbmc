/*
 * This file contains functions to resolve dynamic binding
 * when a clang::MemberExpr refers to a virtual/overriding method, e.g.:
 *  x->F
 *  where F represents a virtual/overriding method
 *
 *  A clang::MemberExpr may be written with `->` or `.`. Both need dynamic
 *  binding when Clang reports the call performs virtual dispatch (e.g. a call
 *  through a reference or a dereferenced pointer, `(*x).F` / `ref.F`). Only an
 *  explicitly-qualified call (`obj.Base::F()`) is devirtualised; a plain
 *  complete-object call still goes through the vtable, which resolves to the
 *  object's own type.
 */

#include <util/base/compiler_defs.h>
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
#include <util/expr/expr_util.h>

bool clang_cpp_convertert::perform_virtual_dispatch(
  const clang::MemberExpr &member)
{
  const clang::Decl &decl = *member.getMemberDecl();
  switch (decl.getKind())
  {
  // TODO: dtor might be virtual too
  // case clang::Decl::CXXDestructor:
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

  const auto md = llvm::dyn_cast<clang::CXXMethodDecl>(member.getMemberDecl());
  assert(md);

  // Then deal with X@vtable_pointer dereferencing
  exprt vtable_ptr_deref;
  get_vft_binding_expr_vtable_ptr(*md, vtable_ptr_deref, base_deref);

  // Then deal with F dereferencing
  exprt f_expr;
  if (get_vft_binding_expr_function(*md, f_expr, vtable_ptr_deref))
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

  // For `ptr->F`, the base is a pointer and the object is `*ptr`. For `obj.F`
  // (a reference or dereferenced-pointer object expression that still needs
  // dynamic dispatch), the base already denotes the object lvalue, so binding
  // its vtable pointer must not add a further dereference.
  if (member.isArrow())
    new_expr = dereference_exprt(base, base.type());
  else
    new_expr = base;

  return false;
}

void clang_cpp_convertert::get_vft_binding_expr_vtable_ptr(
  const clang::CXXMethodDecl &md,
  exprt &new_expr,
  const exprt &base_deref)
{
  std::string base_class_id, base_class_name;
  get_decl_name(*md.getParent(), base_class_name, base_class_id);

  std::string vtable_type_symb_id = vtable_type_prefix + base_class_id;
  typet vtable_type = symbol_typet(vtable_type_symb_id);

  std::string vtable_ptr_name = base_class_id + "::" + vtable_ptr_suffix;
  pointer_typet member_type(vtable_type);
  member_exprt deref_member(base_deref, vtable_ptr_name, member_type);

  // we've got the deref type and member. Now we are ready to make the deref
  // new_expr
  new_expr = dereference_exprt(deref_member, member_type);
}

bool clang_cpp_convertert::get_vft_binding_expr_function(
  const clang::CXXMethodDecl &md,
  exprt &new_expr,
  const exprt &vtable_ptr_deref)
{
  std::string base_class_id, base_class_name;
  get_decl_name(*md.getParent(), base_class_name, base_class_id);

  const symbolt *symbol;
  if (get_method_declaration(md, symbol))
    return true;

  /*
   * This is the component name as in vtable's type symbol
   * e.g. virtual_table::tag.Bird::c:@S@Bird@F@do_something#
   */
  std::string member_comp_name =
    vtable_type_prefix + base_class_id + "::" + symbol->id.as_string();
  pointer_typet member_type(symbol->get_type());
  member_exprt deref_member(vtable_ptr_deref, member_comp_name, member_type);

  new_expr = dereference_exprt(deref_member, member_type);

  return false;
}

bool clang_cpp_convertert::get_virtual_member_function_pointer(
  const clang::CXXMethodDecl &md,
  exprt &new_expr)
{
  /*
   * A call through a pointer to a virtual member function is a virtual call
   * ([expr.mptr.oper], [class.virtual]), so `&X::F` points to a function that
   * makes one: virtual_call::F(this, args) returns this->X@vtable_ptr->F(...)
   */
  const symbolt *method;
  if (get_method_declaration(md, method))
    return true;

  const std::string thunk_id = vcall_prefix + method->id.as_string();
  const symbolt *thunk = context.find_symbol(thunk_id);
  if (!thunk)
  {
    symbolt thunk_symb;
    thunk_symb.id = thunk_id;
    thunk_symb.name = vcall_prefix + method->name.as_string();
    thunk_symb.mode = mode;
    thunk_symb.location = method->location;
    thunk_symb.module = method->module;
    thunk_symb.set_type(method->get_type2());
    add_thunk_method_arguments(thunk_symb);

    const code_typet &code_type = to_code_type(thunk_symb.get_type());
    const code_typet::argumentst &args = code_type.arguments();
    const exprt this_expr = symbol_expr(*ns.lookup(args[0].cmt_identifier()));
    exprt vtable_ptr_deref;
    get_vft_binding_expr_vtable_ptr(
      md, vtable_ptr_deref, dereference_exprt(this_expr, this_expr.type()));
    exprt function;
    if (get_vft_binding_expr_function(md, function, vtable_ptr_deref))
      return true;

    side_effect_expr_function_callt call;
    call.function() = function;
    call.type() = code_type.return_type();
    call.location() = method->location;
    for (const auto &arg : args)
      call.arguments().push_back(symbol_expr(*ns.lookup(arg.cmt_identifier())));

    code_blockt body;
    body.location() = method->location;
    if (call.type().is_empty())
    {
      code_expressiont expr(call);
      expr.location() = method->location;
      body.copy_to_operands(expr);
    }
    else
    {
      code_returnt ret;
      ret.return_value() = call;
      ret.location() = method->location;
      body.copy_to_operands(ret);
    }

    const namespacet *old_ns = std::exchange(migrate_namespace_lookup, &ns);
    expr2tc body2;
    migrate_expr(body, body2);
    migrate_namespace_lookup = old_ns;
    thunk_symb.set_value(body2);

    thunk = context.move_symbol_to_context(thunk_symb);
  }

  new_expr = symbol_expr(*thunk);
  return false;
}

bool clang_cpp_convertert::get_unary_operator_expr(
  const clang::UnaryOperator &uniop,
  exprt &new_expr)
{
  const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(uniop.getSubExpr());
  const auto *md =
    dre ? llvm::dyn_cast<clang::CXXMethodDecl>(dre->getDecl()) : nullptr;
  // The thunk cannot forward the arguments of an ellipsis.
  if (
    uniop.getOpcode() != clang::UO_AddrOf || !md ||
    !is_md_virtual_or_overriding(*md) || md->isVariadic())
    return clang_c_convertert::get_unary_operator_expr(uniop, new_expr);

  typet t;
  if (get_type(uniop.getType(), t))
    return true;

  exprt function;
  if (get_virtual_member_function_pointer(*md, function))
    return true;

  new_expr = exprt("address_of", t);
  new_expr.move_to_operands(function);
  return false;
}
