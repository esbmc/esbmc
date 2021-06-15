#include <solidity-ast-frontend/solidity_decl_tracker.h>

void VarDeclTracker::config()
{
  // config order matters! Do NOT change.
  set_decl_kind();

  // set QualType tracker. Note: this should be set after set_decl_kind()
  set_qualtype_tracker();
  // set subtype based on QualType
  if (qualtype_tracker.get_type_class() == SolidityTypes::TypeBuiltin)
  {
    set_qualtype_bt_kind();
  }
  else
  {
    assert(!"should not be here - unsupported qualtype");
  }
}

void VarDeclTracker::set_decl_kind()
{
  assert(decl_kind == SolidityTypes::DeclKindError); // only allowed to set once during config(). If set twice, something is wrong.
  if (!decl_json.contains("nodeType"))
    assert(!"missing \'nodeType\' in JSON AST node");
  decl_kind = SolidityTypes::get_decl_kind(
      decl_json["nodeType"].get<std::string>()
      );
}

void VarDeclTracker::set_qualtype_tracker()
{
  if (decl_kind == SolidityTypes::DeclVar)
  {
    std::string qual_type = decl_json["typeName"]["nodeType"].get<std::string>();
    if (qual_type == "ElementaryTypeName")
    {
      assert(qualtype_tracker.get_type_class() == SolidityTypes::TypeError); // only allowed to set once during config();
      // Solidity's ElementaryTypeName == Clang's clang::Type::Builtin
      qualtype_tracker.set_type_class(SolidityTypes::TypeBuiltin);
    }
    else
    {
      assert(!"should not be here - unsupported qual_type, please add more types");
    }
  }
  else
  {
    assert(!"should not be here - unsupported decl_kind when setting qualtype_tracker");
  }
}

void VarDeclTracker::set_qualtype_bt_kind()
{
  if (decl_kind == SolidityTypes::DeclVar)
  {
    std::string qual_type = decl_json["typeName"]["typeDescriptions"]["typeString"].get<std::string>();
    if (qual_type == "uint8")
    {
      assert(qualtype_tracker.get_bt_kind() == SolidityTypes::BuiltInError); // only allowed to set once during config();
      // Solidity's ElementaryTypeName::uint8 == Clang's clang::BuiltinType::UChar
      qualtype_tracker.set_bt_kind(SolidityTypes::BuiltInUChar);
    }
    else
    {
      assert(!"should not be here - unsupported qual_type, please add more types");
    }
  }
  else
  {
    assert(!"should not be here - unsupported decl_kind when setting qualtype's bt_kind");
  }
}

void VarDeclTracker::print_decl_json()
{
  printf("### decl_json: ###\n");
  std::cout << std::setw(2) << decl_json << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}
