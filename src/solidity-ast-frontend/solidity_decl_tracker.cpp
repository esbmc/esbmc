#include <solidity-ast-frontend/solidity_decl_tracker.h>

/* =======================================================
 * DeclTracker
 * =======================================================
 */
void DeclTracker::print_decl_json()
{
  printf("### decl_json: ###\n");
  std::cout << std::setw(2) << decl_json << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}

// config this tracker based on json values
void DeclTracker::config(std::string ab_path)
{
  assert(absolute_path == ""); // only allowed to set once during config()
  absolute_path = ab_path;

  // config order matters! Do NOT change:
  set_decl_kind();

  // Decl Base Common: set QualType tracker. Note: this should be set after set_decl_kind()
  set_qualtype_tracker();

  // Decl Base Common: set subtype based on QualType
  if (qualtype_tracker.get_type_class() == SolidityTypes::TypeBuiltin)
  {
    set_qualtype_bt_kind();
  }
  else
  {
    assert(!"should not be here - unsupported qualtype");
  }

  // Decl Base Common: set SourceLocationTracler
  set_sl_tracker_line_number();
  set_sl_tracker_file_name();
  assert(!sl_tracker.get_isValid()); // must be invalid initially. Only allowed to set once during config()
  sl_tracker.set_isValid(true);

  // set NamedDeclTracker
  set_named_decl_name();
  set_named_decl_kind();
}

void DeclTracker::set_named_decl_name()
{
  if (decl_kind == SolidityTypes::DeclVar || decl_kind == SolidityTypes::DeclFunction)
  {
    assert(decl_json.contains("name"));
    std::string decl_name = decl_json["name"].get<std::string>();
    assert(nameddecl_tracker.get_name() == ""); // only allowed to set once during config();
    nameddecl_tracker.set_name(decl_name);
  }
  else
  {
    assert(!"should not be here - unsupported decl_kind when setting namedDecl's name");
  }
}

void DeclTracker::set_named_decl_kind()
{
  assert(nameddecl_tracker.get_named_decl_kind() == SolidityTypes::DeclKindError); // only allowed to set once during config();
  if (decl_kind == SolidityTypes::DeclVar)
  {
    nameddecl_tracker.set_named_decl_kind(SolidityTypes::DeclVar);
  }
  else if (decl_kind == SolidityTypes::DeclFunction)
  {
    nameddecl_tracker.set_named_decl_kind(SolidityTypes::DeclFunction);
  }
  else
  {
    assert(!"should not be here - unsupported decl_kind when setting namedDecl's kind");
  }

  assert(!nameddecl_tracker.get_hasIdentifier()); // only allowed to set once during config();
  nameddecl_tracker.set_hasIdentifier(true); // to be used in conjunction with the name above in converter's static std::string get_decl_name()
}

void DeclTracker::set_sl_tracker_line_number()
{
  assert(sl_tracker.get_line_number() == SourceLocationTracker::lineNumberInvalid); // only allowed to set once during config();
  sl_tracker.set_line_number(1); // TODO: Solidity's source location is NOT clear
}

void DeclTracker::set_sl_tracker_file_name()
{
  assert(sl_tracker.get_file_name() == ""); // only allowed to set once during config();
  sl_tracker.set_file_name(absolute_path);
}

void DeclTracker::set_qualtype_tracker()
{
  if (decl_kind == SolidityTypes::DeclVar)
  {
    // for variable decl, we try to get the type of this variable
    assert(decl_json["typeName"].contains("nodeType"));
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
  else if (decl_kind == SolidityTypes::DeclFunction)
  {
    // for function decl, we try to get the return type of this function
    assert(decl_json.contains("returnParameters"));
    unsigned num_parameters = decl_json["returnParameters"]["parameters"].size();
    assert(num_parameters == 0); // hard coded for now. Expecting
    assert(qualtype_tracker.get_type_class() == SolidityTypes::TypeError); // only allowed to set once during config();
    // Solidity contract has no return parameters implies void, which is Clang's clang::BuiltinType::Void
    qualtype_tracker.set_type_class(SolidityTypes::TypeBuiltin);
  }
  else
  {
    assert(!"should not be here - unsupported decl_kind when setting qualtype_tracker");
  }
}

void DeclTracker::set_qualtype_bt_kind()
{
  if (decl_kind == SolidityTypes::DeclVar)
  {
    assert(decl_json["typeName"]["typeDescriptions"].contains("typeString"));
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
  else if (decl_kind == SolidityTypes::DeclFunction)
  {
    unsigned num_parameters = decl_json["returnParameters"]["parameters"].size();
    assert(num_parameters == 0); // hard coded for now. Expecting
    assert(qualtype_tracker.get_bt_kind() == SolidityTypes::BuiltInError); // only allowed to set once during config();
    // Solidity contract has no return parameters implies void, which is Clang's clang::Type::Builtin
    qualtype_tracker.set_bt_kind(SolidityTypes::BuiltinVoid);
  }
  else
  {
    assert(!"should not be here - unsupported decl_kind when setting qualtype's bt_kind");
  }
}

void DeclTracker::set_decl_kind()
{
  assert(decl_kind == SolidityTypes::DeclKindError); // only allowed to set once during config(). If set twice, something is wrong.
  if (!decl_json.contains("nodeType"))
    assert(!"missing \'nodeType\' in JSON AST node");
  decl_kind = SolidityTypes::get_decl_kind(
      decl_json["nodeType"].get<std::string>()
      );
}

/* =======================================================
 * VarDeclTracker
 * =======================================================
 */

VarDeclTracker::~VarDeclTracker()
{
  printf("@@ deleting VarDeclTracker %s...\n", nameddecl_tracker.get_name().c_str());
}

void VarDeclTracker::config(std::string ab_path)
{
  // config order matters! Do NOT change.
  DeclTracker::config(ab_path);

  // set SourceLocationTracler
  sl_tracker.set_isFunctionOrMethod(false); // because this is a VarDecl, not a function declaration

  // set static lifetime, extern, file_local
  assert(storage_class == SolidityTypes::SCError);
  storage_class = SolidityTypes::SC_None; // hard coded, may need to change in the future
  hasExternalStorage = false; // hard coded, may need to change in the future
  isExternallyVisible = true; // hard coded, may need to change in the future
  set_hasGlobalStorage();

  // set init value
  set_hasInit();
}

void VarDeclTracker::set_hasGlobalStorage()
{
  if (decl_kind == SolidityTypes::DeclVar)
  {
    assert(decl_json.contains("stateVariable"));
    hasGlobalStorage = decl_json["stateVariable"].get<bool>();
  }
  else
  {
    assert(!"should not be here - unsupported decl_kind when setting hasGlobalStorage");
  }
}

void VarDeclTracker::set_hasInit()
{
  if (decl_kind == SolidityTypes::DeclVar)
  {
    if (decl_json.contains("value"))
      hasInit = true;
  }
  else
  {
    assert(!"should not be here - unsupported decl_kind when setting hasInit");
  }
}

/* =======================================================
 * FunctionDeclTracker
 * =======================================================
 */
void FunctionDeclTracker::config(std::string ab_path)
{
  // config order matters! Do NOT change.
  DeclTracker::config(ab_path);

  isImplicit = false; // hard coded, may need to change in the future
  set_defined();

  // set SourceLocationTracler
  sl_tracker.set_isFunctionOrMethod(false); // still false for function declaration

  // set static lifetime
  assert(storage_class == SolidityTypes::SCError);
  storage_class = SolidityTypes::SC_None; // hard coded, may need to change in the future

  // set number of parameters
  set_num_params();

  // set function body
  set_has_body();
}

// set isDefined and isADefinition
void FunctionDeclTracker::set_defined()
{
  assert(decl_json.contains("body")); // expect Solidity function has a function body
  unsigned num_stmt = decl_json["body"]["statements"].size();
  assert(num_stmt > 0); // expect the function body is non-empty

  // because decl_json has an non-empty "body" as asserted above
  isDefined = true;
  isADefinition = true;
}

void FunctionDeclTracker::set_num_params()
{
  assert(decl_json["parameters"].contains("parameters")); // expect Solidity function json has parameter array
  assert(num_args == numParamsInvalid); // only allowed to set once
  num_args = decl_json["parameters"]["parameters"].size();
  if (num_args != 0)
    assert(!"unsupported - function with arguments");
}

void FunctionDeclTracker::set_has_body()
{
  // set function body existence flag
  assert(decl_json.contains("body")); // expect Solidity function has a "body" field
  hasBody = decl_json["body"]["statements"].size() > 0 ? true : false;

  // set function body statements
  assert(hasBody);
  assert(stmt == nullptr); // only allowed to set once during config stage
  stmt = new CompoundStmtTracker();
  // Everything is considered as compound statement class, even with one statement
  stmt->set_stmt_class(SolidityTypes::CompoundStmtClass);
}

FunctionDeclTracker::~FunctionDeclTracker()
{
  printf("@@ Deleting function decl tracker %s...\n", nameddecl_tracker.get_name().c_str());
  delete stmt;
}
