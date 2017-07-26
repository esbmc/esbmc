
#include <clang-c-frontend/clang_c_convert.h>
#include <clang/Tooling/Core/QualTypeNames.h>

std::string clang_c_convertert::get_decl_name(
  const clang::NamedDecl &decl)
{
  if(const clang::IdentifierInfo *identifier = decl.getIdentifier())
    return identifier->getName().str();

  std::string name;
  llvm::raw_string_ostream rso(name);
  decl.printName(rso);
  return rso.str();
}

void clang_c_convertert::get_field_name(
  const clang::FieldDecl& fd,
  std::string &name,
  std::string &pretty_name)
{
  name = pretty_name = get_decl_name(fd);

  if(name.empty())
  {
    typet t;
    get_type(fd.getType(), t);

    if(fd.isBitField())
    {
      exprt width;
      get_expr(*fd.getBitWidth(), width);
      t.width(width.cformat());
    }

    name = clang::TypeName::getFullyQualifiedName(fd.getType(), *ASTContext);
    pretty_name = "anon";
  }
}

void clang_c_convertert::get_var_name(
  const clang::VarDecl& vd,
  std::string &name,
  std::string &pretty_name)
{
  pretty_name = get_decl_name(vd);

  if(!vd.isExternallyVisible())
  {
    locationt vd_location;
    get_location_from_decl(vd, vd_location);

    name += get_modulename_from_path(vd_location.file().as_string()) + "::";
  }

  if(vd.getDeclContext()->isFunctionOrMethod())
  {
    const clang::FunctionDecl &funcd =
      static_cast<const clang::FunctionDecl &>(*vd.getDeclContext());

    name += get_decl_name(funcd) + "::";
    name += integer2string(current_scope_var_num++) + "::";
  }

  name += get_decl_name(vd);
}

void clang_c_convertert::get_function_param_name(
  const clang::ParmVarDecl& pd,
  std::string &name,
  std::string &pretty_name)
{
  pretty_name = get_decl_name(pd);

  locationt pd_location;
  get_location_from_decl(pd, pd_location);

  const clang::FunctionDecl &fd =
    static_cast<const clang::FunctionDecl &>(*pd.getParentFunctionOrMethod());

  name = get_modulename_from_path(pd_location.file().as_string()) + "::";
  name += get_decl_name(fd) + "::";
  name += get_decl_name(pd);
}

void clang_c_convertert::get_function_name(
  const clang::FunctionDecl& fd,
  std::string &name,
  std::string &pretty_name)
{
  name = pretty_name = get_decl_name(fd);

  if(!fd.isExternallyVisible())
  {
    locationt fd_location;
    get_location_from_decl(fd, fd_location);

    name = get_modulename_from_path(fd_location.file().as_string());
    name += "::" + pretty_name;
  }
}

void clang_c_convertert::get_tag_name(
  const clang::RecordDecl& rd,
  std::string &name,
  std::string &pretty_name)
{
  pretty_name =
    clang::TypeName::getFullyQualifiedName(
      ASTContext->getTagDeclType(&rd),
      *ASTContext);

  name = "tag-" + pretty_name;
}
