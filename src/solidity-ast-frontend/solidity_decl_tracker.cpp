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

  unsigned num_stmt = 0;
  // item is like { "0" : { the_expr }}
  for (const auto& item : stmt_json.items())
  {
    /*
    if (std::stoi(item.key()) >= 3) // TODO: remove debug when proceeding with Part IV!
      continue;
    */
    //printf("### statement json: ");
    //::print_decl_json(item.value());
    // populate statement vector based on each statement json object
    const auto& stmt = item.value();
    assert(stmt["nodeType"] == "ExpressionStatement"); // expect all statement to be "ExpressionStatement"
    assert(stmt.contains("expression")); // expect ExpressionStatement has an "expression" key
    printf("@@ ### This is stmt[%u] in CompoundStmtTracker ...\n", num_stmt);
    add_statement(stmt["expression"]);
    ++num_stmt;
  }
}

void CompoundStmtTracker::add_statement(const nlohmann::json& expr)
{
  //std::string _type =  expr["nodeType"].get<std::string>();
  // Construct the statemnt object based on the information in expr json object
  if (expr["nodeType"] == "Assignment")
  {
    StmtTracker* stmt = new BinaryOperatorTracker(expr);
    stmt->config();
    statements.push_back(stmt);
  }
  else if (expr["nodeType"] == "FunctionCall")
  {
    //printf("@@DEBUG FunctionCall: "); ::print_decl_json(expr);
    StmtTracker* stmt = new CallExprTracker(expr);
    stmt->config();
    statements.push_back(stmt);
  }
  else
  {
    assert(!"Unimplemented expression nodeType");
  }
}

/* =======================================================
 * CallExprTracker
 * =======================================================
 */
CallExprTracker::~CallExprTracker()
{
  // delete callee
  assert(callee);
  delete callee;

  // delete args
  for (auto arg : call_args)
  {
    assert(arg);
    delete arg;
  }
}

void CallExprTracker::config()
{
  // order is important. Do NOT change!
  set_stmt_class(SolidityTypes::CallExprClass);

  // TODO: in order to do the concept proof, note this part is hard coded based on the RSH as in
  // "assert( (int) ((int)(unsigned)sum > (int)100));"
  callee = new ImplicitCastExprTracker(stmt_json);
  callee->set_stmt_class(SolidityTypes::ImplicitCastExprClass);
  StmtTrackerPtr implicit_cast_ptr = callee;
  auto implicit_cast_tracker = static_cast<ImplicitCastExprTracker*>(implicit_cast_ptr);
  //printf("@@DEBUG - Callee's Implicit: "); ::print_decl_json(implicit_cast_ptr->stmt_json);
  implicit_cast_tracker->set_sub_expr_kind(SolidityTypes::DeclRefExprClass);

  // set QualTypeTracker:
  // TODO: in order to do the concept proof, note this part is hard coded based on the RSH as in
  // "assert( (int) ((int)(unsigned)sum > (int)100));"
  implicit_cast_tracker->set_pointer_qualtype_tracker();

  // populate args
  if (stmt_json["arguments"].size() > 0)
  {
    unsigned num_args = 0;
    for (const auto& item : stmt_json["arguments"].items())
    {
      //printf("@@DEBUG - arg: "); ::print_decl_json(stmt_json);
      printf("@@ ### Adding args[%u] in CallRefExpr ...\n", num_args);
      add_argument(item.value());
      ++num_args;
    }
  }
  else
  {
    // TODO: Need to construct CallRefExpr that has no args. This can be another reason why
    assert(!"Unsupported - CallRefExpr has no arguments");
  }
}

void CallExprTracker::add_argument(const nlohmann::json& expr)
{
  printf("@@DEBUG - args: "); ::print_decl_json(expr);
  std::string node_type = expr["nodeType"].get<std::string>();
  if (node_type == "BinaryOperation")
  {
    StmtTracker* stmt = new BinaryOperatorTracker(expr);
    StmtTrackerPtr bin_op_tracker_ptr = stmt;
    auto bin_op_tracker = static_cast<BinaryOperatorTracker*>(bin_op_tracker_ptr);
    bin_op_tracker->set_binary_op_gt();
    call_args.push_back(stmt);
  }
  else
  {
    assert(!"Unsupported node_type when creating argument tracker");
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
  set_qualtype_tracker();
}

void BinaryOperatorTracker::set_binary_opcode()
{
  assert(binary_opcode == SolidityTypes::BOError); // only allowed to set once during config() stage
  std::string _operator = stmt_json["operator"].get<std::string>();
  binary_opcode = SolidityTypes::get_binary_op_class(_operator);
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
  // can be "leftHandSide" or "leftExpression" depending on the type of RHS
  std::string hs_key = (lor == "LHS") ? "leftHandSide" : "rightHandSide";

  //printf("@@DEBUG-: "); ::print_decl_json(stmt_json);

  // Split "leftHandSide" and "leftExpression"
  // TODO: in order to do the concept proof, note this part is hard coded based on the RSH as in
  // "sum = (int) ( (int)(int)_x + (int)(int)_y );"
  if (stmt_json.contains(hs_key)) // for "leftHandSide"
  {
    assert(stmt_json.contains(hs_key));
    printf("@@ populating BinaryOperatorTracker's %s ...\n", lor.c_str());
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
        assert(!"unimplemented - other data types in set_lhs_rhs when setting BinaryOperatorTracker");
      }
    }
    else if (node_type == "BinaryOperation") // referring to BO_Add as in _x + _ y
    {
      // Conversion of sum = _x + _y : RHS ;
      // TODO SumBinOp: in order to do the concept proof, note this part is hard coded based on the RSH as in
      // "BinaryOpExpr = (unsigned) ( (int)(unsigned)_x + (int)(unsigned)_y )" as in "sum = BinaryOpExpr;"
      // This is the first "(unsigned)" cast in BinaryOpExpr above
      expr_ptr = new ImplicitCastExprTracker(expr_json);
      expr_ptr->set_stmt_class(SolidityTypes::ImplicitCastExprClass);
      StmtTrackerPtr implicit_cast_ptr = expr_ptr;
      auto implicit_cast_tracker = static_cast<ImplicitCastExprTracker*>(implicit_cast_ptr);
      implicit_cast_tracker->set_sub_expr_kind(SolidityTypes::BinaryOperatorClass);
      // we rely on the upstream BinaryOperator type info to set implicit cast type info
      // as part of the implicit Literal conversion
      implicit_cast_tracker->set_expr_type_str(stmt_json["typeDescriptions"]["typeString"]);
      assert(implicit_cast_tracker->get_expr_type_str() == "uint8");
      implicit_cast_tracker->config();
    }
    else
    {
      printf("Unimplemented %s nodeType in BinaryOperatorTracker\n", lor.c_str());
      assert(0);
    }
  }
  else // for "leftExpression"
  {
    hs_key = (lor == "LHS") ? "leftExpression" : "rightExpression";
    assert(stmt_json.contains(hs_key));
    printf("@@ populating BinaryOperatorTracker's %s (%s) ...\n", lor.c_str(), hs_key.c_str());
    const nlohmann::json& expr_json = stmt_json[hs_key];
    std::string node_type = expr_json["nodeType"].get<std::string>();

    if (node_type == "Identifier" && expr_json.contains("referencedDeclaration"))
    {
      // TODO SumBinOp: in order to do the concept proof, note this part is hard coded based on the RSH as in
      // "BinaryOpExpr = (unsigned) ( (int)(unsigned)_x + (int)(unsigned)_y )" as in "sum = BinaryOpExpr;"
      expr_ptr = new ImplicitCastExprTracker(expr_json);
      expr_ptr->set_stmt_class(SolidityTypes::ImplicitCastExprClass);
      StmtTrackerPtr implicit_cast_ptr = expr_ptr;
      auto implicit_cast_tracker = static_cast<ImplicitCastExprTracker*>(implicit_cast_ptr);
      implicit_cast_tracker->set_sub_expr_kind(SolidityTypes::ImplicitCastExprClass);
      // we rely on the upstream BinaryOperator type info to set implicit cast type info
      // as part of the implicit Literal conversion
      implicit_cast_tracker->set_expr_type_str(stmt_json["typeDescriptions"]["typeString"]);
      assert(implicit_cast_tracker->get_expr_type_str() == "uint8"); // TODO Fix: NOT UChar, should be Int!
      implicit_cast_tracker->config();
    }
    else
    {
      assert(!"Unimplemented - leftExpression or rightExpression");
    }
  }
}

void BinaryOperatorTracker::set_qualtype_tracker()
{
  std::string type = stmt_json["typeDescriptions"]["typeString"].get<std::string>();
  if (type == "uint8")
  {
    assert(qualtype_tracker.get_type_class() == SolidityTypes::TypeError); // only allowed to set once during config();
    assert(qualtype_tracker.get_bt_kind() == SolidityTypes::BuiltInError); // only allowed to set once during config();
    // Solidity's uint8 == Clang's clang::BuiltinType::UChar
    qualtype_tracker.set_type_class(SolidityTypes::TypeBuiltin);
    qualtype_tracker.set_bt_kind(SolidityTypes::BuiltInUChar);
  }
  else
  {
    assert(!"unimplemented - other data types when setting qualtype_tracker in BinaryOperatorTracker");
  }
}

void BinaryOperatorTracker::set_binary_op_gt()
{
  // TODO: in order to do the concept proof, note this part is hard coded for
  // "((int)(unsigned)sum > (int)100));" in the "assert" function call
  // Function hard coded for Opcode ">"
  // We need LHS = (int)(unsigned)_sum, RHS = IntegerLiteral

  // Set statement class
  assert(stmt_class == SolidityTypes::StmtClassError); // only allowed to set once when populating the args
  stmt_class = SolidityTypes::BinaryOperatorClass;

  // Set binary opcode
  assert(binary_opcode == SolidityTypes::BOError); // only allowed to set once when populating the args
  std::string _operator = stmt_json["operator"].get<std::string>();
  binary_opcode = SolidityTypes::get_binary_op_class(_operator);

  // Set LHS = (int)(unsigned)_sum
  assert(!"cool");

  // Set RHS = IntegerLiteral

  // TODO: Set qualtype tracker

  // TODO: Set CastKind
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
  else if (_kind == SolidityTypes::BinaryOperatorClass)
  {
    // TODO SumBinOp: in order to do the concept proof, note this part is hard coded based on the RSH as in
    // "sum = (int) ( (int)(unsigned)_x + (int)(unsigned)_y );"
    //printf("@@DEBUG: "); ::print_decl_json(stmt_json);
    sub_expr = new BinaryOperatorTracker(stmt_json); // we also need the same json object to the sub expr
    sub_expr->set_stmt_class(_kind);
    StmtTrackerPtr sub_expr_ptr = sub_expr;
    auto sub_expr_tracker = static_cast<BinaryOperatorTracker*>(sub_expr_ptr);
    sub_expr_tracker->config();
  }
  else if (_kind == SolidityTypes::ImplicitCastExprClass)
  {
    // TODO SumBinOp: in order to do the concept proof, note this part is hard coded based on the RSH as in
    // "(int)(unsigned)_x" or "(int)(unsigned)_y" with unnecessary (int) removed
    //printf("@@DEBUG--: "); ::print_decl_json(stmt_json); // gives the "(int)(int)_x" decl ref

    // We need a DeclRefExprTracker wrapped in a ImplicitCastTracker
    sub_expr = new ImplicitCastExprTracker(stmt_json);
    sub_expr->set_stmt_class(SolidityTypes::ImplicitCastExprClass);
    StmtTrackerPtr implicit_cast_ptr = sub_expr;
    auto implicit_cast_tracker = static_cast<ImplicitCastExprTracker*>(implicit_cast_ptr);
    implicit_cast_tracker->set_cast_decl_ref_expr(); // to build the DeclRefExpr "_x" as in (int)(unsigned)_x
    implicit_cast_tracker->set_implicit_cast_kind_LR();
    QualTypeTracker& double_implicit_cast_qual = implicit_cast_tracker->get_qualtype_tracker_ref();
    double_implicit_cast_qual.set_type_class(SolidityTypes::TypeBuiltin);
    double_implicit_cast_qual.set_bt_kind(SolidityTypes::BuiltinInt); // TODO Fix: NOT Int, should be UChar?

    //assert(!"nice --");
  }
  else if (_kind == SolidityTypes::DeclRefExprClass)
  {
    // TODO: in order to do the concept proof, note this part is hard coded based on the RSH as in
    // "assert( (int) ((int)(unsigned)sum > (int)100));"
    sub_expr = new DeclRefExprTracker(stmt_json["expression"]); // function name .etc is inside "expression" object
    sub_expr->set_stmt_class(SolidityTypes::DeclRefExprClass);
    // set DeclRef ID and kind
    StmtTrackerPtr decl_ptr = sub_expr;
    auto decl_ref_tracker = static_cast<DeclRefExprTracker*>(decl_ptr);

    // set DeclRef id
    assert(decl_ref_tracker->get_decl_ref_id() == DeclRefExprTracker::declRefIdInvalid); // only allowed to set ONCE
    int _id = stmt_json["expression"]["referencedDeclaration"].get<int>();
    if (_id < 0) // to cope with "-3"
    {
      decl_ref_tracker->set_decl_ref_id(DeclRefExprTracker::declRefIdInvalid - 1);
    }
    else
    {
      assert(!"Unexpected positive id");
    }

    // set DeclRef kind
    std::string _type = stmt_json["expression"]["typeDescriptions"]["typeString"].get<std::string>();
    if (_type == "function (bool) pure") // There is a mismatch between Solidity and C here. We want a function pointer
    {
      // TODO: in order to do the concept proof, note this part is hard coded based on the RSH as in
      // "assert( (int) ((int)(unsigned)sum > (int)100));"

      // set NamedDeclTracker:
      // 1. set_named_decl_name
      assert(stmt_json["expression"].contains("name"));
      std::string decl_name = stmt_json["expression"]["name"].get<std::string>();
      assert(decl_ref_tracker->get_nameddecl_tracker().get_name() == ""); // only allowed to set once during config();
      decl_ref_tracker->set_named_decl_name(decl_name);
      assert(!(decl_ref_tracker->get_nameddecl_tracker().get_hasIdentifier())); // only allowed to set once during config();
      decl_ref_tracker->set_named_decl_has_id(true); // to be used in conjunction with the name above in converter's static std::string get_decl_name()
      // 2. set_named_decl_kind
      assert(decl_ref_tracker->get_nameddecl_tracker().get_named_decl_kind() == SolidityTypes::DeclKindError); // only allowed to set once during config()
      decl_ref_tracker->set_named_decl_kind(SolidityTypes::DeclFunction);
      // 3. set decl_ref_kind
      assert(decl_ref_tracker->get_decl_ref_kind() == SolidityTypes::declRefError); // only allowed to set ONCE
      decl_ref_tracker->set_decl_ref_kind(SolidityTypes::ValueDecl);

      // set QualTypeTracker of the embedded decl_ref_tracker:
      decl_ref_tracker->set_qualtype_tracker();
    }
    else
    {
      assert(!"Unsupported data type for ValDecl");
    }
  }
  else
  {
    assert(!"Unimplemented - other data types of sub expr in implicit cast tracker");
  }
}

void ImplicitCastExprTracker::set_implicit_cast_kind_LR()
{
  assert(implicit_cast_kind == SolidityTypes::castKindError); // only allowed to set once during config() stage
  implicit_cast_kind = SolidityTypes::CK_LValueToRValue; // LValueToRValue cast for C symbols: Solidity does NOT have this!
}

void ImplicitCastExprTracker::set_cast_decl_ref_expr()
{
  // Set conversion tracker for casted decl ref expr:
  // TODO SumBinOp: in order to do the concept proof, note this part is hard coded based on the RSH as in
  // "(int)(unsigned)_x" or "(int)(unsigned)_y" with unnecessary (int) removed
  assert(stmt_json.contains("referencedDeclaration")); // hard coded: expect an "identifier"
  sub_expr = new DeclRefExprTracker(stmt_json);
  sub_expr->set_stmt_class(SolidityTypes::DeclRefExprClass);
  // set DeclRef ID and kind
  StmtTrackerPtr decl_ptr = sub_expr;
  auto decl_ref_tracker = static_cast<DeclRefExprTracker*>(decl_ptr);

  // set DeclRef id
  assert(decl_ref_tracker->get_decl_ref_id() == DeclRefExprTracker::declRefIdInvalid); // only allowed to set ONCE
  decl_ref_tracker->set_decl_ref_id(stmt_json["referencedDeclaration"].get<unsigned>());

  // set DeclRef kind
  if (stmt_json["typeDescriptions"]["typeString"].get<std::string>() == "uint8")
  {
    assert(decl_ref_tracker->get_decl_ref_kind() == SolidityTypes::declRefError); // only allowed to set ONCE
    decl_ref_tracker->set_decl_ref_kind(SolidityTypes::ValueDecl);

    // Set qualtupe_tracker in this decl_ref_tracker
    // Solidity's uint8 == Clang's clang::BuiltinType::UChar
    decl_ref_tracker->set_qualtype_tracker_for_casted_declref();
  }
  else
  {
    assert(!"Unsupported data type for ValDecl");
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

void ImplicitCastExprTracker::set_pointer_qualtype_tracker()
{
  // set QualTypeTracker:
  // TODO: in order to do the concept proof, note this part is hard coded based on the RSH as in
  // "assert( (int) ((int)(unsigned)sum > (int)100));"
  assert(qualtype_tracker.get_type_class() == SolidityTypes::TypeError); // only allowed to set once during config();
  assert(qualtype_tracker.get_bt_kind() == SolidityTypes::BuiltInError); // only allowed to set once during config();
  qualtype_tracker.set_type_class(SolidityTypes::Pointer);

  implicit_cast_kind = SolidityTypes::CK_IntegralCast;
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

/* =======================================================
 * DeclRefExprTracker
 * =======================================================
 */
void DeclRefExprTracker::set_named_decl_name(std::string _name)
{
  nameddecl_tracker.set_name(_name);
}

void DeclRefExprTracker::set_named_decl_kind(SolidityTypes::declKind _kind)
{
  nameddecl_tracker.set_named_decl_kind(_kind);
}

void DeclRefExprTracker::set_named_decl_has_id(bool _v)
{
  nameddecl_tracker.set_hasIdentifier(_v);
}

void DeclRefExprTracker::set_qualtype_tracker()
{
  // TODO: in order to do the concept proof, note this part is hard coded based on the RSH as in
  // "assert( (int) ((int)(unsigned)sum > (int)100));"
  assert(qualtype_tracker.get_type_class() == SolidityTypes::TypeError); // only allowed to set once during config();
  assert(qualtype_tracker.get_bt_kind() == SolidityTypes::BuiltInError); // only allowed to set once during config();
  qualtype_tracker.set_type_class(SolidityTypes::FunctionNoProto);

  // set sub_qualtype type and bt_kind
  qualtype_tracker.set_sub_qualtype_class(SolidityTypes::TypeBuiltin);
  qualtype_tracker.set_sub_qualtype_bt_kind(SolidityTypes::BuiltinInt);
}

void DeclRefExprTracker::set_qualtype_tracker_for_casted_declref()
{
  // Set decl_ref tracker for casted decl ref expr: "_x" as in "(int)(unsigned)_x"
  // TODO SumBinOp: in order to do the concept proof, note this part is hard coded based on the RSH as in
  // "(int)(unsigned)_x" or "(int)(unsigned)_y" with unnecessary (int) removed
  assert(qualtype_tracker.get_type_class() == SolidityTypes::TypeError); // only allowed to set once during config();
  assert(qualtype_tracker.get_bt_kind() == SolidityTypes::BuiltInError); // only allowed to set once during config();
  qualtype_tracker.set_type_class(SolidityTypes::TypeBuiltin);
  qualtype_tracker.set_bt_kind(SolidityTypes::BuiltInUChar);
}


