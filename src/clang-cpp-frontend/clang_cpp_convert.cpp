#include "clang_cpp_convert.h"

#include <clang/AST/DeclCXX.h>
#include <clang/AST/DeclFriend.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/StmtCXX.h>
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
    break;

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

bool clang_cpp_convertert::get_struct_union_class(const clang::RecordDecl &rd)
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
  const clang::CXXRecordDecl *cxxrd =
    llvm::dyn_cast<clang::CXXRecordDecl>(&recordd);
  if(cxxrd != nullptr)
  {
    // So this is a CXXRecordDecl, let's check for (virtual) base classes
    for(const auto &decl : cxxrd->bases())
    {
      // The base class is always a CXXRecordDecl
      const clang::CXXRecordDecl *base =
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

template <typename SpecializationDecl>
bool clang_cpp_convertert::get_template_decl_specialization(
  const SpecializationDecl *D,
  bool DumpExplicitInst,
  bool DumpRefOnly,
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
