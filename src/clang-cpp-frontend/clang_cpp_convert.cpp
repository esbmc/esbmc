// Remove warnings from Clang headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-parameter"
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
#pragma GCC diagnostic pop

#include <clang-cpp-frontend/clang_cpp_convert.h>
#include <util/expr_util.h>
#include <util/std_code.h>
#include <util/std_expr.h>

clang_cpp_convertert::clang_cpp_convertert(
  contextt &_context,
  std::vector<std::unique_ptr<clang::ASTUnit>> &_ASTs)
  : clang_c_convertert(_context, _ASTs)
{
}

bool clang_cpp_convertert::get_decl(const clang::Decl &decl, exprt &new_expr)
{
  new_expr = code_skipt();

  switch(decl.getKind())
  {
  case clang::Decl::LinkageSpec:
  {
    const clang::LinkageSpecDecl &lsd =
      static_cast<const clang::LinkageSpecDecl &>(decl);

    for(auto decl : lsd.decls())
      if(get_decl(*decl, new_expr))
        return true;
    break;
  }

  case clang::Decl::CXXRecord:
  {
    const clang::CXXRecordDecl &cxxrd =
      static_cast<const clang::CXXRecordDecl &>(decl);

    if(get_struct_union_class(cxxrd))
      return true;

    break;
  }

  case clang::Decl::CXXConstructor:
  case clang::Decl::CXXMethod:
  case clang::Decl::CXXDestructor:
  case clang::Decl::CXXConversion:
  {
    const clang::CXXMethodDecl &cxxmd =
      static_cast<const clang::CXXMethodDecl &>(decl);

    assert(llvm::dyn_cast<clang::TemplateDecl>(&cxxmd) == nullptr);
    if(get_function(cxxmd, new_expr))
      return true;

    break;
  }

  case clang::Decl::Namespace:
  {
    const clang::NamespaceDecl &namesd =
      static_cast<const clang::NamespaceDecl &>(decl);

    for(auto decl : namesd.decls())
      if(get_decl(*decl, new_expr))
        return true;

    break;
  }

  case clang::Decl::FunctionTemplate:
  {
    const clang::FunctionTemplateDecl &fd =
      static_cast<const clang::FunctionTemplateDecl &>(decl);

    if(get_template_decl(&fd, true, new_expr))
      return true;
    break;
  }

  case clang::Decl::ClassTemplate:
  {
    const clang::ClassTemplateDecl &cd =
      static_cast<const clang::ClassTemplateDecl &>(decl);

    if(get_template_decl(&cd, false, new_expr))
      return true;
    break;
  }

  case clang::Decl::ClassTemplateSpecialization:
  {
    const clang::ClassTemplateSpecializationDecl &cd =
      static_cast<const clang::ClassTemplateSpecializationDecl &>(decl);

    if(get_struct_union_class(cd))
      return true;
    break;
  }

  case clang::Decl::Friend:
  {
    const clang::FriendDecl &fd = static_cast<const clang::FriendDecl &>(decl);

    if(fd.getFriendDecl() != nullptr)
      if(get_decl(*fd.getFriendDecl(), new_expr))
        return true;
    break;
  }

  // We can ignore any these declarations
  case clang::Decl::ClassTemplatePartialSpecialization:
  case clang::Decl::Using:
  case clang::Decl::UsingShadow:
  case clang::Decl::UsingDirective:
  case clang::Decl::TypeAlias:
  case clang::Decl::NamespaceAlias:
  case clang::Decl::AccessSpec:
  case clang::Decl::UnresolvedUsingValue:
  case clang::Decl::UnresolvedUsingTypename:
    break;

  default:
    return clang_c_convertert::get_decl(decl, new_expr);
  }

  return false;
}

bool clang_cpp_convertert::get_type(
  const clang::QualType &q_type,
  typet &new_type)
{
  return clang_c_convertert::get_type(q_type, new_type);
}

bool clang_cpp_convertert::get_type(
  const clang::Type &the_type,
  typet &new_type)
{
  switch(the_type.getTypeClass())
  {
  case clang::Type::SubstTemplateTypeParm:
  {
    const clang::SubstTemplateTypeParmType &substmpltt =
      static_cast<const clang::SubstTemplateTypeParmType &>(the_type);

    if(get_type(substmpltt.getReplacementType(), new_type))
      return true;
    break;
  }

  case clang::Type::TemplateSpecialization:
  {
    const clang::TemplateSpecializationType &templSpect =
      static_cast<const clang::TemplateSpecializationType &>(the_type);

    if(get_type(templSpect.desugar(), new_type))
      return true;
    break;
  }

  case clang::Type::MemberPointer:
  {
    const clang::MemberPointerType &mpt =
      static_cast<const clang::MemberPointerType &>(the_type);

    typet sub_type;
    if(get_type(mpt.getPointeeType(), sub_type))
      return true;

    typet class_type;
    if(get_type(*mpt.getClass(), class_type))
      return true;

    new_type = gen_pointer_type(sub_type);
    new_type.set("to-member", class_type);
    break;
  }

  default:
    return clang_c_convertert::get_type(the_type, new_type);
  }

  return false;
}

bool clang_cpp_convertert::get_var(const clang::VarDecl &vd, exprt &new_expr)
{
  // Only convert instantiated variables
  if(vd.getDeclContext()->isDependentContext())
    return false;

  return clang_c_convertert::get_var(vd, new_expr);
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

bool clang_cpp_convertert::get_struct_union_class(const clang::RecordDecl &rd)
{
  // Only convert instantiated functions/methods
  if(rd.isDependentContext())
    return false;

  return clang_c_convertert::get_struct_union_class(rd);
}

bool clang_cpp_convertert::get_struct_union_class_fields(
  const clang::RecordDecl &rd,
  struct_union_typet &type)
{
  // If a struct is defined inside a extern C, it will be a RecordDecl
  if(auto cxxrd = llvm::dyn_cast<clang::CXXRecordDecl>(&rd))
  {
    // So this is a CXXRecordDecl, let's check for (virtual) base classes
    for(const auto &decl : cxxrd->bases())
    {
      // The base class is always a CXXRecordDecl
      const clang::CXXRecordDecl *base =
        decl.getType().getTypePtr()->getAsCXXRecordDecl();
      assert(base != nullptr);

      // First, parse the fields
      for(auto const &field : base->fields())
      {
        // We don't add if private
        if(field->getAccess() >= clang::AS_private)
          continue;

        struct_typet::componentt comp;
        if(get_decl(*field, comp))
          return true;

        // Don't add fields that have global storage (e.g., static)
        if(const clang::VarDecl *nd = llvm::dyn_cast<clang::VarDecl>(field))
          if(nd->hasGlobalStorage())
            continue;

        type.components().push_back(comp);
      }
    }
  }

  return clang_c_convertert::get_struct_union_class_fields(rd, type);
}

bool clang_cpp_convertert::get_struct_union_class_methods(
  const clang::RecordDecl &recordd,
  struct_union_typet &)
{
  // If a struct is defined inside a extern C, it will be a RecordDecl
  const clang::CXXRecordDecl *cxxrd =
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

bool clang_cpp_convertert::get_expr(const clang::Stmt &stmt, exprt &new_expr)
{
  locationt location;
  get_start_location_from_stmt(stmt, location);

  switch(stmt.getStmtClass())
  {
  case clang::Stmt::CXXReinterpretCastExprClass:
    // TODO: ReinterpretCast should actually generate a bitcast
  case clang::Stmt::CXXFunctionalCastExprClass:
  case clang::Stmt::CXXStaticCastExprClass:
  case clang::Stmt::CXXConstCastExprClass:
  {
    const clang::CastExpr &cast = static_cast<const clang::CastExpr &>(stmt);

    if(get_cast_expr(cast, new_expr))
      return true;

    break;
  }

  case clang::Stmt::CXXDefaultArgExprClass:
  {
    const clang::CXXDefaultArgExpr &cxxdarg =
      static_cast<const clang::CXXDefaultArgExpr &>(stmt);

    if(get_expr(*cxxdarg.getExpr(), new_expr))
      return true;

    break;
  }

  case clang::Stmt::CXXDynamicCastExprClass:
  {
    const clang::CXXDynamicCastExpr &cast =
      static_cast<const clang::CXXDynamicCastExpr &>(stmt);

    if(cast.isAlwaysNull())
    {
      typet t;
      if(get_type(cast.getType(), t))
        return true;

      new_expr = gen_zero(gen_pointer_type(t));
    }
    else if(get_cast_expr(cast, new_expr))
      return true;

    break;
  }

  case clang::Stmt::CXXBoolLiteralExprClass:
  {
    const clang::CXXBoolLiteralExpr &bool_literal =
      static_cast<const clang::CXXBoolLiteralExpr &>(stmt);

    if(bool_literal.getValue())
      new_expr = true_exprt();
    else
      new_expr = false_exprt();
    break;
  }

  case clang::Stmt::CXXMemberCallExprClass:
  {
    const clang::CXXMemberCallExpr &member_call =
      static_cast<const clang::CXXMemberCallExpr &>(stmt);

    const clang::Stmt *callee = member_call.getCallee();

    exprt callee_expr;
    if(get_expr(*callee, callee_expr))
      return true;

    typet type;
    if(get_type(member_call.getType(), type))
      return true;

    side_effect_expr_function_callt call;
    call.function() = callee_expr;
    call.type() = type;

    // Add implicit object call: a this pointer or an object
    exprt implicit_object;
    if(get_expr(*member_call.getImplicitObjectArgument(), implicit_object))
      return true;

    call.arguments().push_back(implicit_object);

    // Do args
    for(const clang::Expr *arg : member_call.arguments())
    {
      exprt single_arg;
      if(get_expr(*arg, single_arg))
        return true;

      call.arguments().push_back(single_arg);
    }

    new_expr = call;
    break;
  }

  case clang::Stmt::CXXOperatorCallExprClass:
  {
    const clang::CXXOperatorCallExpr &operator_call =
      static_cast<const clang::CXXOperatorCallExpr &>(stmt);

    const clang::Stmt *callee = operator_call.getCallee();

    exprt callee_expr;
    if(get_expr(*callee, callee_expr))
      return true;

    typet type;
    if(get_type(operator_call.getType(), type))
      return true;

    side_effect_expr_function_callt call;
    call.function() = callee_expr;
    call.type() = type;

    // Do args
    for(const clang::Expr *arg : operator_call.arguments())
    {
      exprt single_arg;
      if(get_expr(*arg, single_arg))
        return true;

      call.arguments().push_back(single_arg);
    }

    new_expr = call;
    break;
  }

  case clang::Stmt::ExprWithCleanupsClass:
  {
    const clang::ExprWithCleanups &ewc =
      static_cast<const clang::ExprWithCleanups &>(stmt);

    if(get_expr(*ewc.getSubExpr(), new_expr))
      return true;

    break;
  }

  case clang::Stmt::CXXBindTemporaryExprClass:
  {
    const clang::CXXBindTemporaryExpr &cxxbtmp =
      static_cast<const clang::CXXBindTemporaryExpr &>(stmt);

    if(get_expr(*cxxbtmp.getSubExpr(), new_expr))
      return true;

    break;
  }

  case clang::Stmt::SubstNonTypeTemplateParmExprClass:
  {
    const clang::SubstNonTypeTemplateParmExpr &substnttp =
      static_cast<const clang::SubstNonTypeTemplateParmExpr &>(stmt);

    if(get_expr(*substnttp.getReplacement(), new_expr))
      return true;

    break;
  }

  case clang::Stmt::MaterializeTemporaryExprClass:
  {
    const clang::MaterializeTemporaryExpr &mtemp =
      static_cast<const clang::MaterializeTemporaryExpr &>(stmt);

    if(get_expr(*mtemp.GetTemporaryExpr(), new_expr))
      return true;

    break;
  }

  case clang::Stmt::CXXNewExprClass:
  {
    const clang::CXXNewExpr &ne = static_cast<const clang::CXXNewExpr &>(stmt);

    typet t;
    if(get_type(ne.getType(), t))
      return true;

    if(ne.isArray())
    {
      new_expr = side_effect_exprt("cpp_new[]", t);

      // TODO: Implement support when the array size is empty
      assert(ne.getArraySize().hasValue());
      exprt size;
      if(get_expr(*(ne.getArraySize().getValue()), size))
        return true;

      new_expr.size(size);
    }
    else
    {
      new_expr = side_effect_exprt("cpp_new", t);
    }

    if(ne.hasInitializer())
    {
      exprt lhs("new_object", t);
      lhs.cmt_lvalue(true);

      exprt init;
      if(get_expr(*ne.getInitializer(), init))
        return true;

      convert_expression_to_code(init);

      new_expr.initializer(init);
    }

    break;
  }

  case clang::Stmt::CXXPseudoDestructorExprClass:
  {
    new_expr = exprt("pseudo_destructor");
    break;
  }

  case clang::Stmt::CXXScalarValueInitExprClass:
  {
    const clang::CXXScalarValueInitExpr &cxxsvi =
      static_cast<const clang::CXXScalarValueInitExpr &>(stmt);

    typet t;
    if(get_type(cxxsvi.getType(), t))
      return true;

    new_expr = gen_zero(t);
    break;
  }

  case clang::Stmt::TypeTraitExprClass:
  {
    const clang::TypeTraitExpr &tt =
      static_cast<const clang::TypeTraitExpr &>(stmt);

    if(tt.getValue())
      new_expr = true_exprt();
    else
      new_expr = false_exprt();
    break;
  }

  default:
    return clang_c_convertert::get_expr(stmt, new_expr);
  }

  new_expr.location() = location;
  return false;
}

template <typename SpecializationDecl>
bool clang_cpp_convertert::get_template_decl_specialization(
  const SpecializationDecl *D,
  bool DumpExplicitInst,
  bool,
  exprt &new_expr)
{
  for(auto *redecl_with_bad_type : D->redecls())
  {
    auto *redecl = llvm::dyn_cast<SpecializationDecl>(redecl_with_bad_type);
    if(!redecl)
    {
      assert(
        llvm::isa<clang::CXXRecordDecl>(redecl_with_bad_type) &&
        "expected an injected-class-name");
      continue;
    }

    switch(redecl->getTemplateSpecializationKind())
    {
    case clang::TSK_ExplicitInstantiationDeclaration:
    case clang::TSK_ExplicitInstantiationDefinition:
    case clang::TSK_ExplicitSpecialization:
      if(!DumpExplicitInst)
        break;
      // Fall through.
    case clang::TSK_Undeclared:
    case clang::TSK_ImplicitInstantiation:
      if(get_decl(*redecl, new_expr))
        return true;
      break;
    }
  }

  return false;
}

template <typename TemplateDecl>
bool clang_cpp_convertert::get_template_decl(
  const TemplateDecl *D,
  bool DumpExplicitInst,
  exprt &new_expr)
{
  for(auto *Child : D->specializations())
    if(get_template_decl_specialization(
         Child, DumpExplicitInst, !D->isCanonicalDecl(), new_expr))
      return true;

  return false;
}
