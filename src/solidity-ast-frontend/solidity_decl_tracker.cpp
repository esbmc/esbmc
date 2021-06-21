#include <solidity-ast-frontend/solidity_decl_tracker.h>

/* =======================================================
 * DeclTracker
 * =======================================================
 */
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
    assert(!"Unimplemented decl_kind when setting qualtype_tracker");
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
  // set id
  set_id();
}

void VarDeclTracker::set_id()
{
  if (decl_kind == SolidityTypes::DeclVar)
  {
    assert(decl_json.contains("id"));
    assert(id == idInvalid); // only allowed to set once
    id = decl_json["id"].get<unsigned>();
  }
  else
  {
    assert(!"should not be here - unsupported decl_kind when setting id for var decl");
  }
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
  unsigned num_stmt = decl_json["body"]["statements"].size();
  hasBody = num_stmt > 0 ? true : false;

  // set function body statements
  assert(hasBody);
  assert(stmt == nullptr); // only allowed to set once during config stage

  // Everything is considered as compound statement class, even with just one statement
  stmt = new CompoundStmtTracker(decl_json["body"]["statements"]);
  stmt->config();
}

FunctionDeclTracker::~FunctionDeclTracker()
{
  printf("@@ Deleting function decl tracker %s...\n", nameddecl_tracker.get_name().c_str());
  assert(stmt);
  delete stmt;
}

/* =======================================================
 * CompoundStmtTracker
 * =======================================================
 */
CompoundStmtTracker::~CompoundStmtTracker()
{
  for (auto stmt : statements)
  {
    assert(stmt);
    delete stmt;
  }
}

void CompoundStmtTracker::config()
{
  set_stmt_class(SolidityTypes::CompoundStmtClass);

  // item is like { "0" : { the_expr }}
  for (const auto& item : stmt_json.items())
  {
    if (std::stoi(item.key()) >= 2) // TODO: remove debug when proceeding with Part III and IV!
      continue;
    // populate statement vector based on each statement json object
    const auto& stmt = item.value();
    assert(stmt["nodeType"] == "ExpressionStatement"); // expect all statement to be "ExpressionStatement"
    assert(stmt.contains("expression")); // expect ExpressionStatement has an "expression" key
    add_statement(stmt["expression"]);
  }
}

void CompoundStmtTracker::add_statement(const nlohmann::json& expr)
{
  // Construct the statemnt object based on the information in expr json object
  if (expr["nodeType"] == "Assignment")
  {
    StmtTracker* stmt = new BinaryOperatorTracker(expr);
    stmt->config();
    statements.push_back(stmt);
  }
  else
  {
    assert(!"Unimplemented expression nodeType");
  }
}

/* =======================================================
 * BinaryOperatorTracker
 * =======================================================
 */
BinaryOperatorTracker::~BinaryOperatorTracker()
{
  assert(lhs);
  delete lhs;
  assert(rhs);
  delete rhs;
}

void BinaryOperatorTracker::config()
{
  // order is important. Do NOT change!
  set_stmt_class(SolidityTypes::BinaryOperatorClass);
  set_binary_opcode();
  set_lhs();
  set_rhs();
}

void BinaryOperatorTracker::set_binary_opcode()
{
  assert(binary_opcode == SolidityTypes::BOError); // only allowed to set once during config() stage
  binary_opcode = SolidityTypes::get_binary_op_class(
      stmt_json["operator"].get<std::string>()
      );
}

void BinaryOperatorTracker::set_lhs()
{
  set_lhs_or_rhs(lhs, "LHS");
}

void BinaryOperatorTracker::set_rhs()
{
  set_lhs_or_rhs(rhs, "RHS");
}

void BinaryOperatorTracker::set_lhs_or_rhs(StmtTrackerPtr& expr_ptr, std::string lor)
{
  assert(!expr_ptr); // only allowed to set once during config() stage

  assert((lor == "LHS") || (lor == "RHS"));
  std::string hs_key = (lor == "LHS") ? "leftHandSide" : "rightHandSide";
  printf("@@ populating BinaryOperatorTracker's %s ...\n", lor.c_str());

  assert(stmt_json.contains(hs_key)); // expect such json object to have a LHS key
  const nlohmann::json& expr_json = stmt_json[hs_key];
  std::string node_type = expr_json["nodeType"].get<std::string>();

  if (node_type == "Identifier" && expr_json.contains("referencedDeclaration"))
  {
    // Solidity expr's "Identifier" /\ has "referencedDeclaration" --->
    //    clang::Stmt::DeclRefExprClass
    expr_ptr = new DeclRefExprTracker(expr_json);
    expr_ptr->set_stmt_class(SolidityTypes::DeclRefExprClass);
    // set DeclRef ID and kind
    StmtTrackerPtr decl_ptr = expr_ptr;
    auto decl_ref_tracker = static_cast<DeclRefExprTracker*>(decl_ptr);

    // set DeclRef id
    assert(decl_ref_tracker->get_decl_ref_id() == DeclRefExprTracker::declRefIdInvalid); // only allowed to set ONCE
    decl_ref_tracker->set_decl_ref_id(expr_json["referencedDeclaration"].get<unsigned>());

    // set DeclRef kind
    if (expr_json["typeDescriptions"]["typeString"].get<std::string>() == "uint8")
    {
      assert(decl_ref_tracker->get_decl_ref_kind() == SolidityTypes::declRefError); // only allowed to set ONCE
      decl_ref_tracker->set_decl_ref_kind(SolidityTypes::ValueDecl);
    }
    else
    {
      assert(!"Unsupported data type for ValDecl");
    }
  }
  else if (node_type == "Literal")
  {
    std::string type_str = expr_json["typeDescriptions"]["typeString"].get<std::string>();
    if (type_str.find("int_const") != std::string::npos)
    {
      // Solidity expr's "Literal" /\ "typeString" has "int_" --->
      //      clang::Stmt::IntegerLiteralClass wrapped in clang::Stmt::ImplicitCastExprClass
      expr_ptr = new ImplicitCastExprTracker(expr_json);
      expr_ptr->set_stmt_class(SolidityTypes::ImplicitCastExprClass);
      StmtTrackerPtr implicit_cast_ptr = expr_ptr;
      auto implicit_cast_tracker = static_cast<ImplicitCastExprTracker*>(implicit_cast_ptr);
      implicit_cast_tracker->set_sub_expr_kind(SolidityTypes::IntegerLiteralClass);
      // we rely on the upstream BinaryOperator type info to set implicit cast type info
      // as part of the implicit Literal conversion
      implicit_cast_tracker->set_expr_type_str(stmt_json["typeDescriptions"]["typeString"]);
      assert(implicit_cast_tracker->get_expr_type_str() == "uint8");
      implicit_cast_tracker->config();
    }
    else
    {
      assert(!"unimplemented - other data types");
    }
  }
  else
  {
    printf("Unimplemented %s nodeType in BinaryOperatorTracker\n", lor.c_str());
    assert(0);
  }
}

/* =======================================================
 * ImplicitCastExprTracker
 * =======================================================
 */
ImplicitCastExprTracker::~ImplicitCastExprTracker()
{
  assert(sub_expr);
  delete sub_expr;
}

void ImplicitCastExprTracker::set_sub_expr_kind(SolidityTypes::stmtClass _kind)
{
  if (_kind == SolidityTypes::IntegerLiteralClass)
  {
    sub_expr = new IntegerLiteralTracker(stmt_json); // we also need the same json object to the sub expr
    sub_expr->set_stmt_class(_kind);
    StmtTrackerPtr sub_expr_ptr = sub_expr;
    auto sub_expr_tracker = static_cast<IntegerLiteralTracker*>(sub_expr_ptr);
    sub_expr_tracker->config();
  }
  else
  {
    assert(!"Unimplemented - other data types of sub expr in implicit cast tracker");
  }
}

void ImplicitCastExprTracker::config()
{
  // order is important. Do NOT change!
  // Decl Base Common: set QualType tracker.
  set_qualtype_tracker();
  set_implicit_cast_kind();
}

void ImplicitCastExprTracker::set_implicit_cast_kind()
{
  if (expr_type_str == "uint8")
  {
    assert(implicit_cast_kind == SolidityTypes::castKindError); // only allowed to set once during config() stage
    implicit_cast_kind = SolidityTypes::CK_IntegralCast;
  }
  else
  {
    assert(!"Unimplemented data types when setting implicit cast kind in ImplicitCastExprTracker");
  }
}

void ImplicitCastExprTracker::set_qualtype_tracker()
{
  if (expr_type_str == "uint8")
  {
    assert(qualtype_tracker.get_type_class() == SolidityTypes::TypeError); // only allowed to set once during config();
    assert(qualtype_tracker.get_bt_kind() == SolidityTypes::BuiltInError); // only allowed to set once during config();
    // Solidity's uint8 == Clang's clang::BuiltinType::UChar
    qualtype_tracker.set_type_class(SolidityTypes::TypeBuiltin);
    qualtype_tracker.set_bt_kind(SolidityTypes::BuiltInUChar);
  }
  else
  {
    assert(!"Unimplemented data types when setting qualtype_tracker in ImplicitCastExprTracker");
  }
}

/* =======================================================
 * IntegerLiteralTracker
 * =======================================================
 */
void IntegerLiteralTracker::config()
{
  // order is important. Do NOT change!
  // Decl Base Common: set QualType tracker.
  set_qualtype_tracker();
}

void IntegerLiteralTracker::set_qualtype_tracker()
{
  assert(stmt_class == SolidityTypes::IntegerLiteralClass);
  // for variable decl, we try to get the type of this variable
  assert(stmt_json["typeDescriptions"].contains("typeString"));
  std::string qual_type = stmt_json["typeDescriptions"]["typeString"].get<std::string>();
  if (qual_type.find("int_const") != std::string::npos)
  {
    assert(qualtype_tracker.get_type_class() == SolidityTypes::TypeError); // only allowed to set once during config();
    assert(qualtype_tracker.get_bt_kind() == SolidityTypes::BuiltInError); // only allowed to set once during config();
    // Solidity expr RHS contains "int_const" == Clang's clang::Type::Builtin
    qualtype_tracker.set_type_class(SolidityTypes::TypeBuiltin);
    qualtype_tracker.set_bt_kind(SolidityTypes::BuiltinInt);
    // set sign-extended integer value
    assert(sgn_ext_value == sgnExtValInvalid);
    assert(stmt_json.contains("value"));
    //sgn_ext_value = stmt_json["value"].get<int64_t>(); // Suprisingly, this does NOT work!
    std::string val_str = stmt_json["typeDescriptions"]["typeString"].get<std::string>();
    if (val_str.find_last_of(' ') != std::string::npos)
    {
      std::string value = val_str.substr((val_str.find_last_of(' ') + 1));
      sgn_ext_value = (int64_t)(std::stoi(value));
    }
    else
    {
      assert(!"Unable to extract value from type string");
    }
  }
  else
  {
    assert(!"Unimplemented data types in IntegerLiteralTracker");
  }
}
