
#include <ansi-c/type2name.h>
#include <clang-c-frontend/clang_c_convert.h>

void clang_c_convertert::get_field_name(
  const clang::FieldDecl& fd,
  std::string &name,
  std::string &pretty_name)
{
  name = pretty_name = fd.getName().str();

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

    name += funcd.getName().str() + "::";
    name += integer2string(current_scope_var_num++) + "::";
  }

  name += vd.getName().str();
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
  name += fd.getName().str() + "::";
  name += pd.getName().str();
}

void clang_c_convertert::get_function_name(
  const clang::FunctionDecl& fd,
  std::string &base_name,
  std::string &pretty_name)
{
  base_name = pretty_name = fd.getName().str();

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
  name = rd.getName().str();
  if(!name.empty())
    return false;

  // Try to get the name from typedef (if one exists)
  if (const clang::TagDecl *tag = llvm::dyn_cast<clang::TagDecl>(&rd))
  {
    if (const clang::TypedefNameDecl *tnd = rd.getTypedefNameForAnonDecl())
    {
      name = tnd->getName().str();
      return false;
    }
    else if (tag->getIdentifier())
    {
      name = tag->getName().str();
      return false;
    }
  }

  struct_union_typet t;
  if(rd.isStruct())
    t = struct_typet();
  else if(rd.isUnion())
    t = union_typet();
  else
    // This should never be reached
    abort();

  clang::RecordDecl *record_def = rd.getDefinition();
  if(get_struct_union_class_fields(*record_def, t))
    return true;

  name = type2name(t);
  return false;
}
