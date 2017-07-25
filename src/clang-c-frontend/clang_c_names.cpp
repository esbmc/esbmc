
#include <clang-c-frontend/clang_c_convert.h>
#include <util/type2name.h>

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

    name = type2name(t);
    pretty_name = "anon";
  }
}

void clang_c_convertert::get_var_name(
  const clang::VarDecl& vd,
  std::string& name)
{
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
  std::string& name)
{
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
  std::string &base_name,
  std::string &pretty_name)
{
  base_name = pretty_name = get_decl_name(fd);

  if(!fd.isExternallyVisible())
  {
    locationt fd_location;
    get_location_from_decl(fd, fd_location);

    pretty_name = get_modulename_from_path(fd_location.file().as_string());
    pretty_name += "::" + base_name;
  }
}

bool clang_c_convertert::get_tag_name(
  const clang::RecordDecl& rd,
  std::string &name)
{
  name = get_decl_name(rd);
  if(!name.empty())
    return false;

  // Try to get the name from typedef (if one exists)
  if (const clang::TagDecl *tag = llvm::dyn_cast<clang::TagDecl>(&rd))
  {
    if (const clang::TypedefNameDecl *tnd = rd.getTypedefNameForAnonDecl())
    {
      name = get_decl_name(*tnd);
      return false;
    }
    else if (tag->getIdentifier())
    {
      name = get_decl_name(*tag);
      return false;
    }
  }

  struct_union_typet t;
  if(rd.isUnion())
    t = union_typet();
  else
    t = struct_typet();

  clang::RecordDecl *record_def = rd.getDefinition();
  if(get_struct_union_class_fields(*record_def, t))
    return true;

  name = type2name(t);
  return false;
}
