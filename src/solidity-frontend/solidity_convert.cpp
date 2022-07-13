#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/typecast.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/mp_arith.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <regex>

#include <fstream>

solidity_convertert::solidity_convertert(
  contextt &_context,
  nlohmann::json &_ast_json,
  const std::string &_sol_func,
  const std::string &_contract_path)
  : context(_context),
    ns(context),
    ast_json(_ast_json),
    sol_func(_sol_func),
    contract_path(_contract_path),
    global_scope_id(0),
    current_scope_var_num(1),
    current_functionDecl(nullptr),
    current_forStmt(nullptr),
    current_functionName("")
{
  std::ifstream in(_contract_path);
  contract_contents.assign(
    (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool solidity_convertert::convert()
{
  // This function consists of two parts:
  //  1. First, we perform pattern-based verificaiton
  //  2. Then we populate the context with symbols annotated based on the each AST node, and hence prepare for the GOTO conversion.

  if(!ast_json.contains(
       "nodes")) // check json file contains AST nodes as Solidity might change
    assert(!"JSON file does not contain any AST nodes");

  if(
    !ast_json.contains(
      "absolutePath")) // check json file contains AST nodes as Solidity might change
    assert(!"JSON file does not contain absolutePath");

  absolute_path = ast_json["absolutePath"].get<std::string>();

  // By now the context should have the symbols of all ESBMC's intrinsics and the dummy main
  // We need to convert Solidity AST nodes to the equivalent symbols and add them to the context
  nlohmann::json nodes = ast_json["nodes"];

  bool found_contract_def = false;
  unsigned index = 0;
  for(nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end();
      ++itr, ++index)
  {
    // ignore the meta information and locate nodes in ContractDefinition
    std::string node_type = (*itr)["nodeType"].get<std::string>();
    if(node_type == "ContractDefinition") // contains AST nodes we need
    {
      global_scope_id = (*itr)["id"];
      found_contract_def = true;
      // pattern-based verification
      assert(itr->contains("nodes"));
      auto pattern_check =
        std::make_unique<pattern_checker>((*itr)["nodes"], sol_func);
      if(pattern_check->do_pattern_check())
        return true; // 'true' indicates something goes wrong.
    }
  }
  assert(found_contract_def);

  // reasoning-based verification
  index = 0;
  for(nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end();
      ++itr, ++index)
  {
    std::string node_type = (*itr)["nodeType"].get<std::string>();
    if(node_type == "ContractDefinition") // rule source-unit
    {
      if(convert_ast_nodes(*itr))
        return true; // 'true' indicates something goes wrong.
    }
  }

  return false; // 'false' indicates successful completion.
}

bool solidity_convertert::convert_ast_nodes(const nlohmann::json &contract_def)
{
  unsigned index = 0;
  nlohmann::json ast_nodes = contract_def["nodes"];
  for(nlohmann::json::iterator itr = ast_nodes.begin(); itr != ast_nodes.end();
      ++itr, ++index)
  {
    nlohmann::json ast_node = *itr;
    std::string node_name = ast_node["name"].get<std::string>();
    std::string node_type = ast_node["nodeType"].get<std::string>();
    log_debug(
      "@@ Converting node[{}]: name={}, nodeType={} ...",
      index,
      node_name.c_str(),
      node_type.c_str());
    exprt dummy_decl;
    if(get_decl(ast_node, dummy_decl))
      return true;
  }

  // After converting all AST nodes, current_functionDecl should be restored to nullptr.
  assert(current_functionDecl == nullptr);

  return false;
}

bool solidity_convertert::get_decl(
  const nlohmann::json &ast_node,
  exprt &new_expr)
{
  new_expr = code_skipt();

  if(!ast_node.contains("nodeType"))
    assert(!"Missing \'nodeType\' filed in ast_node");

  SolidityGrammar::ContractBodyElementT type =
    SolidityGrammar::get_contract_body_element_t(ast_node);

  // based on each element as in Solidty grammar "rule contract-body-element"
  switch(type)
  {
  case SolidityGrammar::ContractBodyElementT::StateVarDecl:
  {
    return get_var_decl(ast_node, new_expr); // rule state-variable-declaration
  }
  case SolidityGrammar::ContractBodyElementT::FunctionDef:
  {
    return get_function_definition(ast_node); // rule function-definition
  }
  default:
  {
    assert(!"Unimplemented type in rule contract-body-element");
    return true;
  }
  }

  return false;
}

bool solidity_convertert::get_var_decl_stmt(
  const nlohmann::json &ast_node,
  exprt &new_expr)
{
  // For rule variable-declaration-statement
  new_expr = code_skipt();

  if(!ast_node.contains("nodeType"))
    assert(!"Missing \'nodeType\' filed in ast_node");

  SolidityGrammar::VarDeclStmtT type =
    SolidityGrammar::get_var_decl_stmt_t(ast_node);
  log_debug(
    "	@@@ got Variable-declaration-statement: "
    "SolidityGrammar::VarDeclStmtT::{}",
    SolidityGrammar::var_decl_statement_to_str(type));

  switch(type)
  {
  case SolidityGrammar::VarDeclStmtT::VariableDecl:
  {
    return get_var_decl(ast_node, new_expr); // rule variable-declaration
  }
  default:
  {
    assert(!"Unimplemented type in rule variable-declaration-statement");
    return true;
  }
  }

  return false;
}

// rule state-variable-declaration
// rule variable-declaration-statement
bool solidity_convertert::get_var_decl(
  const nlohmann::json &ast_node,
  exprt &new_expr)
{
  // For Solidity rule state-variable-declaration:
  // 1. populate typet
  typet t;
  // VariableDeclaration node contains both "typeName" and "typeDescriptions".
  // However, ExpressionStatement node just contains "typeDescriptions".
  // For consistensy, we use ["typeName"]["typeDescriptions"] as in state-variable-declaration
  // to improve the re-usability of get_type* function, when dealing with non-array var decls.
  // For array, do NOT use ["typeName"]. Otherwise, it will cause problem
  // when populating typet in get_cast
  bool dyn_array = is_dyn_array(ast_node["typeDescriptions"]);
  if(dyn_array)
  {
    // append size expr in typeDescription JSON object
    const nlohmann::json &type_descriptor = add_dyn_array_size_expr(
      ast_node["typeName"]["typeDescriptions"], ast_node);
    if(get_type_description(type_descriptor, t))
      return true;
  }
  else
  {
    if(get_type_description(ast_node["typeName"]["typeDescriptions"], t))
      return true;
  }

  bool is_state_var = ast_node["stateVariable"] == true;

  // 2. populate id and name
  std::string name, id;
  if(is_state_var)
    get_state_var_decl_name(ast_node, name, id);
  else if(current_functionDecl)
  {
    assert(current_functionName != "");
    get_var_decl_name(ast_node, name, id);
  }
  else
  {
    assert(
      !"Error: ESBMC could not find the parent scope for this local variable");
  }

  // 3. populate location
  locationt location_begin;
  get_location_from_decl(ast_node, location_begin);

  // 4. populate debug module name
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());

  // 5. set symbol attributes
  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, t, name, id, location_begin);

  symbol.lvalue = true;
  symbol.static_lifetime = is_state_var;
  symbol.file_local = !is_state_var;
  symbol.is_extern = false;

  // For state var decl, we look for "value".
  // For local var decl, we look for "initialValue"
  bool has_init =
    (ast_node.contains("value") || ast_node.contains("initialValue")) &&
    !dyn_array;
  if(symbol.static_lifetime && !symbol.is_extern && !has_init)
  {
    symbol.value = gen_zero(t, true);
    symbol.value.zero_initializer(true);
  }

  // 6. add symbol into the context
  // just like clang-c-frontend, we have to add the symbol before converting the initial assignment
  symbolt &added_symbol = *move_symbol_to_context(symbol);

  // 7. populate init value if there is any
  code_declt decl(symbol_expr(added_symbol));

  if(has_init)
  {
    nlohmann::json init_value =
      is_state_var ? ast_node["value"] : ast_node["initialValue"];

    nlohmann::json int_literal_type = nullptr;

    auto expr_type = SolidityGrammar::get_expression_t(init_value);
    bool expr_is_literal = expr_type == SolidityGrammar::Literal;
    bool expr_is_un_op = expr_type == SolidityGrammar::UnaryOperatorClass;

    auto subexpr = init_value["subExpression"];
    bool subexpr_is_literal = subexpr == nullptr
                                ? false
                                : SolidityGrammar::get_expression_t(subexpr) ==
                                    SolidityGrammar::Literal;
    if(expr_is_literal || (expr_is_un_op && subexpr_is_literal))
      int_literal_type = ast_node["typeDescriptions"];

    exprt val;
    if(get_expr(init_value, int_literal_type, val))
      return true;

    solidity_gen_typecast(ns, val, t);

    added_symbol.value = val;
    decl.operands().push_back(val);
  }

  decl.location() = location_begin;

  new_expr = decl;

  return false;
}

bool solidity_convertert::get_function_definition(
  const nlohmann::json &ast_node)
{
  // For Solidity rule function-definition:
  // Order matters! do not change!
  // 1. Check fd.isImplicit() --- skipped since it's not applicable to Solidity
  // 2. Check fd.isDefined() and fd.isThisDeclarationADefinition()
  if(
    !ast_node
      ["implemented"]) // TODO: for interface function, it's just a definition. Add something like "&& isInterface_JustDefinition()"
    return false;

  // Check intrinsic functions
  if(check_intrinsic_function(ast_node))
    return false;

  // 3. Set current_scope_var_num, current_functionDecl and old_functionDecl
  current_scope_var_num = 1;
  const nlohmann::json *old_functionDecl = current_functionDecl;
  current_functionDecl = &ast_node;
  current_functionName = (*current_functionDecl)["name"].get<std::string>();

  // 4. Return type
  code_typet type;
  if(get_type_description(ast_node["returnParameters"], type.return_type()))
    return true;

  // 5. Check fd.isVariadic(), fd.isInlined()
  //  Skipped since Solidity does not support variadic (optional args) or inline function.
  //  Actually "inline" doesn not make sense in Solidity

  // 6. Populate "locationt location_begin"
  locationt location_begin;
  get_location_from_decl(ast_node, location_begin);

  // 7. Populate "std::string id, name"
  std::string name, id;
  get_function_definition_name(ast_node, name, id);

  if(name == "func_dynamic")
    printf("@@ found func_dynamic\n");

  // 8. populate "std::string debug_modulename"
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());

  // 9. Populate "symbol.static_lifetime", "symbol.is_extern" and "symbol.file_local"
  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, type, name, id, location_begin);

  symbol.lvalue = true;
  symbol.is_extern =
    false; // TODO: hard coded for now, may need to change later
  symbol.file_local = false;

  // 10. Add symbol into the context
  symbolt &added_symbol = *move_symbol_to_context(symbol);

  // 11. Convert parameters, if no parameter, assume ellipis
  //  - Convert params before body as they may get referred by the statement in the body
  SolidityGrammar::ParameterListT params =
    SolidityGrammar::get_parameter_list_t(ast_node["parameters"]);
  if(params == SolidityGrammar::ParameterListT::EMPTY)
  {
    // assume ellipsis if the function has no parameters
    type.make_ellipsis();
  }
  else
  {
    // convert parameters if the function has them
    // update the typet, since typet contains parameter annotations
    unsigned num_param_decl = 0;
    for(const auto &decl : ast_node["parameters"]["parameters"].items())
    {
      const nlohmann::json &func_param_decl = decl.value();

      code_typet::argumentt param;
      if(get_function_params(func_param_decl, param))
        return true;

      type.arguments().push_back(param);
      ++num_param_decl;
    }
    log_debug("  @@@ number of param decls: {}", num_param_decl);
  }

  added_symbol.type = type;

  // 12. Convert body and embed the body into the same symbol
  if(ast_node.contains("body"))
  {
    exprt body_exprt;
    get_block(ast_node["body"], body_exprt);

    added_symbol.value = body_exprt;
  }

  //assert(!"done - finished all expr stmt in function?");

  // 13. Restore current_functionDecl
  current_functionDecl =
    old_functionDecl; // for __ESBMC_assume, old_functionDecl == null

  return false;
}

bool solidity_convertert::get_function_params(
  const nlohmann::json &pd,
  exprt &param)
{
  // 1. get parameter type
  typet param_type;
  if(get_type_description(pd["typeDescriptions"], param_type))
    return true;

  // 2. check array: array-to-pointer decay
  bool is_array = SolidityGrammar::get_type_name_t(pd["typeDescriptions"]);
  if(is_array)
  {
    assert(!"Unimplemented - funciton parameter is array type");
  }

  // 3a. get id and name
  std::string id, name;
  assert(current_functionName != ""); // we are converting a function param now
  assert(current_functionDecl);
  get_var_decl_name(pd, name, id);

  param = code_typet::argumentt();
  param.type() = param_type;
  param.cmt_base_name(name);

  // 3b. check name empty: Not applicable to Solidity.
  // In Solidity, a function definition should also have parameters names
  assert(name != "");

  // 4. get location
  locationt location_begin;
  get_location_from_decl(pd, location_begin);

  param.cmt_identifier(id);
  param.location() = location_begin;

  // 5. get symbol
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());
  symbolt param_symbol;
  get_default_symbol(
    param_symbol, debug_modulename, param_type, name, id, location_begin);

  // 6. set symbol's lvalue, is_parameter and file local
  param_symbol.lvalue = true;
  param_symbol.is_parameter = true;
  param_symbol.file_local = true;

  // 7. check if function is defined: Not applicable to Solidity.
  assert((*current_functionDecl).contains("body"));

  // 8. add symbol to the context
  move_symbol_to_context(param_symbol);
  return false;
}

bool solidity_convertert::get_block(
  const nlohmann::json &block,
  exprt &new_expr)
{
  // For rule block
  locationt location;
  get_start_location_from_stmt(block, location);

  SolidityGrammar::BlockT type = SolidityGrammar::get_block_t(block);
  log_debug(
    "	@@@ got Block: SolidityGrammar::BlockT::{}",
    SolidityGrammar::block_to_str(type));

  switch(type)
  {
  // equivalent to clang::Stmt::CompoundStmtClass
  // deal with a block of statements
  case SolidityGrammar::BlockT::Statement:
  {
    const nlohmann::json &stmts = block["statements"];

    code_blockt _block;
    unsigned ctr = 0;
    // items() returns a key-value pair with key being the index
    for(auto const &stmt_kv : stmts.items())
    {
      exprt statement;
      if(get_statement(stmt_kv.value(), statement))
        return true;

      convert_expression_to_code(statement);
      _block.operands().push_back(statement);
      ++ctr;
    }
    log_debug(" \t@@@ CompoundStmt has {} statements", ctr);

    locationt location_end;
    get_final_location_from_stmt(block, location_end);

    _block.end_location(location_end);

    new_expr = _block;
    break;
  }
  default:
  {
    assert(!"Unimplemented type in rule block");
    return true;
  }
  }

  new_expr.location() = location;
  return false;
}

bool solidity_convertert::get_statement(
  const nlohmann::json &stmt,
  exprt &new_expr)
{
  // For rule statement
  // Since this is an additional layer of grammar rules compared to clang C, we do NOT set location here.
  // Just pass the new_expr reference to the next layer.

  SolidityGrammar::StatementT type = SolidityGrammar::get_statement_t(stmt);
  log_debug(
    "	@@@ got Stmt: SolidityGrammar::StatementT::{}",
    SolidityGrammar::statement_to_str(type));

  switch(type)
  {
  case SolidityGrammar::StatementT::Block:
  {
    get_block(stmt, new_expr);
    break;
  }
  case SolidityGrammar::StatementT::ExpressionStatement:
  {
    get_expr(stmt["expression"], new_expr);
    break;
  }
  case SolidityGrammar::StatementT::VariableDeclStatement:
  {
    const nlohmann::json &declgroup = stmt["declarations"];

    codet decls("decl-block");
    unsigned ctr = 0;
    // N.B. Although Solidity AST JSON uses "declarations": [],
    // the size of this array is alway 1!
    // A second declaration will become another stmt in "statements" array
    // e.g. "statements" : [
    //  {"declarations": [], "id": 1}
    //  {"declarations": [], "id": 2}
    //  {"declarations": [], "id": 3}
    // ]
    for(const auto &it : declgroup.items())
    {
      // deal with local var decl with init value
      nlohmann::json decl = it.value();
      if(stmt.contains("initialValue"))
      {
        // Need to combine init value with the decl JSON object
        decl["initialValue"] = stmt["initialValue"];
      }

      exprt single_decl;
      if(get_var_decl_stmt(decl, single_decl))
        return true;

      decls.operands().push_back(single_decl);
      ++ctr;
    }
    log_debug(" \t@@@ DeclStmt group has {} decls", ctr);

    new_expr = decls;
    break;
  }
  case SolidityGrammar::StatementT::ReturnStatement:
  {
    if(!current_functionDecl)
      assert(!"Error: ESBMC could not find the parent scope for this ReturnStatement");

    // 1. get return type
    // TODO: Fix me! Assumptions:
    //  a). It's "return <expr>;" not "return;"
    //  b). <expr> is pointing to a DeclRefExpr, we need to wrap it in an ImplicitCastExpr as a subexpr
    assert(stmt.contains("expression"));
    assert(stmt["expression"].contains("referencedDeclaration"));

    typet return_type;
    if(get_type_description(
         stmt["expression"]["typeDescriptions"], return_type))
      return true;

    // 2. get return value
    code_returnt ret_expr;
    const nlohmann::json &rtn_expr = stmt["expression"];
    // wrap it in an ImplicitCastExpr to convert LValue to RValue
    nlohmann::json implicit_cast_expr =
      make_implicit_cast_expr(rtn_expr, "LValueToRValue");

    exprt val;
    if(get_expr(implicit_cast_expr, val))
      return true;

    solidity_gen_typecast(ns, val, return_type);
    ret_expr.return_value() = val;

    new_expr = ret_expr;
    break;
  }
  case SolidityGrammar::StatementT::ForStatement:
  {
    // Based on rule for-statement
    // TODO: Fix me. Assuming for loop contains everything
    assert(
      stmt.contains("initializationExpression") && stmt.contains("condition") &&
      stmt.contains("loopExpression") && stmt.contains("body"));
    // TODO: Fix me. Currently we don't support nested for loop
    assert(current_forStmt == nullptr);
    current_forStmt = &stmt;

    // 1. annotate init
    codet init =
      code_skipt(); // code_skipt() means no init in for-stmt, e.g. for (; i< 10; ++i)
    if(get_statement(stmt["initializationExpression"], init))
      return true;

    convert_expression_to_code(init);

    // 2. annotate condition
    exprt cond = true_exprt();
    if(get_expr(stmt["condition"], cond))
      return true;

    // 3. annotate increment
    codet inc = code_skipt();
    if(get_statement(stmt["loopExpression"], inc))
      return true;

    convert_expression_to_code(inc);

    // 4. annotate body
    codet body = code_skipt();
    if(get_statement(stmt["body"], body))
      return true;

    convert_expression_to_code(body);

    code_fort code_for;
    code_for.init() = init;
    code_for.cond() = cond;
    code_for.iter() = inc;
    code_for.body() = body;

    new_expr = code_for;
    break;
  }
  case SolidityGrammar::StatementT::IfStatement:
  {
    // Based on rule if-statement
    // 1. Condition: make a exprt for condition
    exprt cond;
    if(get_expr(stmt["condition"], cond))
      return true;

    // 2. Then: make a exprt for trueBody
    exprt then;
    if(get_statement(stmt["trueBody"], then))
      return true;

    convert_expression_to_code(then);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(cond, then);

    // 3. Else: make a exprt for "falseBody" if the if-statement node contains an "else" block
    if(stmt.contains("falseBody"))
    {
      exprt else_expr;
      if(get_statement(stmt["falseBody"], else_expr))
        return true;

      convert_expression_to_code(else_expr);
      if_expr.copy_to_operands(else_expr);
    }

    new_expr = if_expr;
    break;
  }
  case SolidityGrammar::StatementT::WhileStatement:
  {
    exprt cond = true_exprt();
    if(get_expr(stmt["condition"], cond))
      return true;

    codet body = codet();
    if(get_block(stmt["body"], body))
      return true;

    convert_expression_to_code(body);

    code_whilet code_while;
    code_while.cond() = cond;
    code_while.body() = body;

    new_expr = code_while;
    break;
  }
  default:
  {
    assert(!"Unimplemented type in rule statement");
    return true;
  }
  }

  return false;
}

/**
 * @brief Populate the out parameter with the expression based on
 * the solidity expression grammar
 *
 * @param expr The expression ast is to be converted to the IR
 * @param new_expr Out parameter to hold the conversion
 * @return true iff the conversion has failed
 * @return false iff the conversion was successful
 */
bool solidity_convertert::get_expr(const nlohmann::json &expr, exprt &new_expr)
{
  return get_expr(expr, nullptr, new_expr);
}

/**
 * @brief Populate the out parameter with the expression based on
 * the solidity expression grammar
 *
 * @param expr The expression that is to be converted to the IR
 * @param int_literal_type Type information ast to create the the literal
 * type in the IR (only needed for when the expression is a literal)
 * @param new_expr Out parameter to hold the conversion
 * @return true iff the conversion has failed
 * @return false iff the conversion was successful
 */
bool solidity_convertert::get_expr(
  const nlohmann::json &expr,
  const nlohmann::json &int_literal_type,
  exprt &new_expr)
{
  // For rule expression
  // We need to do location settings to match clang C's number of times to set the locations when recurring
  locationt location;
  get_start_location_from_stmt(expr, location);

  SolidityGrammar::ExpressionT type = SolidityGrammar::get_expression_t(expr);
  log_debug(
    "	@@@ got Expr: SolidityGrammar::ExpressionT::{}",
    SolidityGrammar::expression_to_str(type));

  switch(type)
  {
  case SolidityGrammar::ExpressionT::BinaryOperatorClass:
  {
    if(get_binary_operator_expr(expr, new_expr))
      return true;

    break;
  }
  case SolidityGrammar::ExpressionT::UnaryOperatorClass:
  {
    if(get_unary_operator_expr(expr, int_literal_type, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::DeclRefExprClass:
  {
    if(expr["referencedDeclaration"] > 0)
    {
      // Soldity uses +ve odd numbers to refer to var or functions declared in the contract
      const nlohmann::json &decl = find_decl_ref(expr["referencedDeclaration"]);

      if(!check_intrinsic_function(decl))
      {
        if(decl["nodeType"] == "VariableDeclaration")
        {
          if(get_var_decl_ref(decl, new_expr))
            return true;
        }
        else if(decl["nodeType"] == "FunctionDefinition")
        {
          if(get_func_decl_ref(decl, new_expr))
            return true;
        }
        else
        {
          assert(!"Unsupported DeclRefExprClass type");
        }
      }
      else
      {
        // for special functions, we need to deal with it separately
        if(get_decl_ref_builtin(expr, new_expr))
          return true;
      }
    }
    else
    {
      // Soldity uses -ve odd numbers to refer to built-in var or functions that
      // are NOT declared in the contract
      if(get_decl_ref_builtin(expr, new_expr))
        return true;
    }

    break;
  }
  case SolidityGrammar::ExpressionT::Literal:
  {
    // make a type-name json for integer literal conversion
    std::string the_value = expr["value"].get<std::string>();
    const nlohmann::json &literal = expr["typeDescriptions"];
    SolidityGrammar::ElementaryTypeNameT type_name =
      SolidityGrammar::get_elementary_type_name_t(literal);
    log_debug(
      "	@@@ got Literal: SolidityGrammar::ElementaryTypeNameT::{}",
      SolidityGrammar::elementary_type_name_to_str(type_name));

    switch(type_name)
    {
    case SolidityGrammar::ElementaryTypeNameT::INT_LITERAL:
    {
      assert(int_literal_type != nullptr);
      if(convert_integer_literal(int_literal_type, the_value, new_expr))
        return true;
      break;
    }
    case SolidityGrammar::ElementaryTypeNameT::BOOL:
    {
      if(convert_bool_literal(literal, the_value, new_expr))
        return true;
      break;
    }
    case SolidityGrammar::ElementaryTypeNameT::STRING_LITERAL:
    {
      if(convert_string_literal(the_value, new_expr))
        return true;
      break;
    }
    default:
      assert(!"Literal not implemented");
    }

    break;
  }
  case SolidityGrammar::ExpressionT::CallExprClass:
  {
    // 1. Get callee expr
    const nlohmann::json &callee_expr_json = expr["expression"];

    // wrap it in an ImplicitCastExpr to perform conversion of FunctionToPointerDecay
    nlohmann::json implicit_cast_expr =
      make_implicit_cast_expr(callee_expr_json, "FunctionToPointerDecay");
    exprt callee_expr;
    if(get_expr(implicit_cast_expr, callee_expr))
      return true;

    // 2. Get type
    // Need to "decrypt" the typeDescriptions and manually make a typeDescription
    nlohmann::json callee_rtn_type =
      make_callexpr_return_type(callee_expr_json["typeDescriptions"]);
    typet type;
    if(get_type_description(callee_rtn_type, type))
      return true;

    side_effect_expr_function_callt call;
    call.function() = callee_expr;
    call.type() = type;

    // 3. Set side_effect_expr_function_callt
    unsigned num_args = 0;
    for(const auto &arg : expr["arguments"].items())
    {
      exprt single_arg;
      if(get_expr(arg.value(), single_arg))
        return true;

      call.arguments().push_back(single_arg);
      ++num_args;
    }
    log_debug("  @@ num_args={}", num_args);

    // 4. Convert call arguments
    new_expr = call;
    break;
  }
  case SolidityGrammar::ExpressionT::ImplicitCastExprClass:
  {
    if(get_cast_expr(expr, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::IndexAccess:
  {
    // 1. get type, this is the base type of array
    typet t;
    if(get_type_description(expr["typeDescriptions"], t))
      return true;

    // 2. get the decl ref of the array
    // wrap it in an ImplicitCastExpr to perform conversion of ArrayToPointerDecay
    nlohmann::json implicit_cast_expr =
      make_implicit_cast_expr(expr["baseExpression"], "ArrayToPointerDecay");

    exprt array;
    if(get_expr(implicit_cast_expr, array))
      return true;

    // 3. get the position index
    exprt pos;
    if(get_expr(expr["indexExpression"], expr["typeDescriptions"], pos))
      return true;

    new_expr = index_exprt(array, pos, t);
    break;
  }
  default:
  {
    assert(!"Unimplemented type in rule expression");
    return true;
  }
  }

  new_expr.location() = location;
  return false;
}

bool solidity_convertert::get_binary_operator_expr(
  const nlohmann::json &expr,
  exprt &new_expr)
{
  // preliminary step for recursive BinaryOperation
  current_BinOp_type.push(&(expr["typeDescriptions"]));

  // 1. Convert LHS and RHS
  // For "Assignment" expression, it's called "leftHandSide" or "rightHandSide".
  // For "BinaryOperation" expression, it's called "leftExpression" or "leftExpression"
  exprt lhs, rhs;
  if(expr.contains("leftHandSide"))
  {
    nlohmann::json literalType = expr["leftHandSide"]["typeDescriptions"];

    if(get_expr(expr["leftHandSide"], lhs))
      return true;

    if(get_expr(expr["rightHandSide"], literalType, rhs))
      return true;
  }
  else if(expr.contains("leftExpression"))
  {
    nlohmann::json commonType = expr["commonType"];

    if(get_expr(expr["leftExpression"], commonType, lhs))
      return true;

    if(get_expr(expr["rightExpression"], commonType, rhs))
      return true;
  }
  else
    assert(!"should not be here - unrecognized LHS and RHS keywords in expression JSON");

  // 2. Get type
  typet t;
  assert(current_BinOp_type.size());
  const nlohmann::json &binop_type = *(current_BinOp_type.top());
  if(get_type_description(binop_type, t))
    return true;

  // 3. Convert opcode
  SolidityGrammar::ExpressionT opcode =
    SolidityGrammar::get_expr_operator_t(expr);
  log_debug(
    "	@@@ got binop.getOpcode: SolidityGrammar::{}",
    SolidityGrammar::expression_to_str(opcode));
  switch(opcode)
  {
  case SolidityGrammar::ExpressionT::BO_Assign:
  {
    new_expr = side_effect_exprt("assign", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Add:
  {
    if(t.is_floatbv())
      assert(!"Solidity does not support FP arithmetic as of v0.8.6.");
    else
      new_expr = exprt("+", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Sub:
  {
    if(t.is_floatbv())
      assert(!"Solidity does not support FP arithmetic as of v0.8.6.");
    else
      new_expr = exprt("-", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_GT:
  {
    new_expr = exprt(">", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_LT:
  {
    new_expr = exprt("<", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_NE:
  {
    new_expr = exprt("notequal", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_EQ:
  {
    new_expr = exprt("=", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Rem:
  {
    new_expr = exprt("mod", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_LAnd:
  {
    new_expr = exprt("and", t);
    break;
  }
  default:
  {
    assert(!"Unimplemented binary operator");
  }
  }

  // 4. Copy to operands
  new_expr.copy_to_operands(lhs, rhs);

  // Pop current_BinOp_type.push as we've finished this conversion
  current_BinOp_type.pop();

  return false;
}

bool solidity_convertert::get_unary_operator_expr(
  const nlohmann::json &expr,
  const nlohmann::json &int_literal_type,
  exprt &new_expr)
{
  // TODO: Fix me! Currently just support prefix == true,e.g. pre-increment
  assert(expr["prefix"]);

  // 1. get UnaryOperation opcode
  SolidityGrammar::ExpressionT opcode =
    SolidityGrammar::get_unary_expr_operator_t(expr, expr["prefix"]);
  log_debug(
    "	@@@ got uniop.getOpcode: SolidityGrammar::{}",
    SolidityGrammar::expression_to_str(opcode));

  // 2. get type
  typet uniop_type;
  if(get_type_description(expr["typeDescriptions"], uniop_type))
    return true;

  // 3. get subexpr
  exprt unary_sub;
  if(get_expr(expr["subExpression"], int_literal_type, unary_sub))
    return true;

  switch(opcode)
  {
  case SolidityGrammar::ExpressionT::UO_PreDec:
  {
    new_expr = side_effect_exprt("predecrement", uniop_type);
    break;
  }
  case SolidityGrammar::ExpressionT::UO_PreInc:
  {
    new_expr = side_effect_exprt("preincrement", uniop_type);
    break;
  }
  case SolidityGrammar::ExpressionT::UO_Minus:
  {
    new_expr = exprt("unary-", uniop_type);
    break;
  }
  default:
  {
    assert(!"Unimplemented unary operator");
  }
  }

  new_expr.operands().push_back(unary_sub);
  return false;
}

bool solidity_convertert::get_cast_expr(
  const nlohmann::json &cast_expr,
  exprt &new_expr)
{
  // 1. convert subexpr
  exprt expr;
  if(get_expr(cast_expr["subExpr"], expr))
    return true;

  // 2. get type
  typet type;
  if(cast_expr["castType"].get<std::string>() == "ArrayToPointerDecay")
  {
    // Array's cast_expr will have cast_expr["subExpr"]["typeDescriptions"]:
    //  "typeIdentifier": "t_array$_t_uint8_$2_memory_ptr"
    //  "typeString": "uint8[2] memory"
    // For the data above, SolidityGrammar::get_type_name_t will return ArrayTypeName.
    // But we want Pointer type. Hence, adjusting the type manually to make it like:
    //   "typeIdentifier": "ArrayToPtr",
    //   "typeString": "uint8[2] memory"
    nlohmann::json adjusted_type =
      make_array_to_pointer_type(cast_expr["subExpr"]["typeDescriptions"]);
    if(get_type_description(adjusted_type, type))
      return true;
  }
  else
  {
    if(get_type_description(cast_expr["subExpr"]["typeDescriptions"], type))
      return true;
  }

  // 3. get cast type and generate typecast
  SolidityGrammar::ImplicitCastTypeT cast_type =
    SolidityGrammar::get_implicit_cast_type_t(
      cast_expr["castType"].get<std::string>());
  switch(cast_type)
  {
  case SolidityGrammar::ImplicitCastTypeT::LValueToRValue:
  {
    solidity_gen_typecast(ns, expr, type);
    break;
  }
  case SolidityGrammar::ImplicitCastTypeT::FunctionToPointerDecay:
  case SolidityGrammar::ImplicitCastTypeT::ArrayToPointerDecay:
  {
    break;
  }
  default:
  {
    assert(!"Unimplemented implicit cast type");
  }
  }

  new_expr = expr;
  return false;
}

bool solidity_convertert::get_var_decl_ref(
  const nlohmann::json &decl,
  exprt &new_expr)
{
  // Function to configure new_expr that has a +ve referenced id, referring to a variable declaration
  assert(decl["nodeType"] == "VariableDeclaration");
  std::string name, id;
  if(decl["stateVariable"])
    get_state_var_decl_name(decl, name, id);
  else
    get_var_decl_name(decl, name, id);

  typet type;
  if(get_type_description(
       decl["typeName"]["typeDescriptions"],
       type)) // "type-name" as in state-variable-declaration
    return true;

  new_expr = exprt("symbol", type);
  new_expr.identifier(id);
  new_expr.cmt_lvalue(true);
  new_expr.name(name);
  return false;
}

bool solidity_convertert::get_func_decl_ref(
  const nlohmann::json &decl,
  exprt &new_expr)
{
  // Function to configure new_expr that has a +ve referenced id, referring to a function declaration
  assert(decl["nodeType"] == "FunctionDefinition");
  std::string name, id;
  get_function_definition_name(decl, name, id);

  typet type;
  if(get_func_decl_ref_type(
       decl, type)) // "type-name" as in state-variable-declaration
    return true;

  new_expr = exprt("symbol", type);
  new_expr.identifier(id);
  new_expr.cmt_lvalue(true);
  new_expr.name(name);
  return false;
}

bool solidity_convertert::get_decl_ref_builtin(
  const nlohmann::json &decl,
  exprt &new_expr)
{
  // Function to configure new_expr that has a -ve referenced id
  // -ve ref id means built-in functions or variables.
  // Add more special function names here
  assert(
    decl["name"] == "assert" || decl["name"] == "__ESBMC_assume" ||
    decl["name"] == "__VERIFIER_assume");

  std::string name, id;
  name = decl["name"].get<std::string>();
  id = name;

  // manually unrolled recursion here
  // type config for Builtin && Int
  typet type;
  // Creat a new code_typet, parse the return_type and copy the code_typet to typet
  code_typet convert_type;
  typet return_type;
  if(
    decl["name"] == "assert" || decl["name"] == "__ESBMC_assume" ||
    decl["name"] == "__VERIFIER_assume")
  {
    assert(decl["typeDescriptions"]["typeString"] == "function (bool) pure");
    // clang's assert(.) uses "signed_int" as assert(.) type (NOT the argument type),
    // while Solidity's assert uses "bool" as assert(.) type (NOT the argument type).
    return_type = bool_type();
    std::string c_type = "bool";
    return_type.set("#cpp_type", c_type);
    convert_type.return_type() = return_type;

    if(!convert_type.arguments().size())
      convert_type.make_ellipsis();
  }
  else
  {
    assert(!"Unsupported special functions");
  }

  type = convert_type;

  new_expr = exprt("symbol", type);
  new_expr.identifier(id);
  new_expr.cmt_lvalue(true);
  new_expr.name(name);
  return false;
}

bool solidity_convertert::get_type_description(
  const nlohmann::json &type_name,
  typet &new_type)
{
  // For Solidity rule type-name:
  SolidityGrammar::TypeNameT type = SolidityGrammar::get_type_name_t(type_name);

  switch(type)
  {
  case SolidityGrammar::TypeNameT::ElementaryTypeName:
  {
    // rule state-variable-declaration
    return get_elementary_type_name(type_name, new_type);
  }
  case SolidityGrammar::TypeNameT::ParameterList:
  {
    // rule parameter-list
    // Used for Solidity function parameter or return list
    return get_parameter_list(type_name, new_type);
  }
  case SolidityGrammar::TypeNameT::Pointer:
  {
    // auxiliary type: pointer (FuncToPtr decay)
    // This part is for FunctionToPointer decay only
    assert(
      type_name["typeString"].get<std::string>().find("function") !=
      std::string::npos);

    // Since Solidity does not have this, first make a pointee
    nlohmann::json pointee = make_pointee_type(type_name);
    typet sub_type;
    if(get_func_decl_ref_type(pointee, sub_type))
      return true;

    if(sub_type.is_struct() || sub_type.is_union())
      assert(!"struct or union is NOT supported");

    new_type = gen_pointer_type(sub_type);
    break;
  }
  case SolidityGrammar::TypeNameT::PointerArrayToPtr:
  {
    // auxiliary type: pointer (FuncToPtr decay)
    // This part is for FunctionToPointer decay only
    assert(
      type_name["typeIdentifier"].get<std::string>().find("ArrayToPtr") !=
      std::string::npos);

    // Array type descriptor is like:
    //  "typeIdentifier": "ArrayToPtr",
    //  "typeString": "uint8[2] memory"

    // Since Solidity does not have this, first make a pointee
    typet sub_type;
    if(get_array_to_pointer_type(type_name, sub_type))
      return true;

    if(
      sub_type.is_struct() ||
      sub_type.is_union()) // for "assert(sum > 100)", false || false
      assert(!"struct or union is NOT supported");

    new_type = gen_pointer_type(sub_type);
    break;
  }
  case SolidityGrammar::TypeNameT::ArrayTypeName:
  {
    // Deal with array with constant size, e.g., int a[2]; Similar to clang::Type::ConstantArray
    // array's typeDescription is in a compact form, e.g.:
    //    "typeIdentifier": "t_array$_t_uint8_$2_storage_ptr",
    //    "typeString": "uint8[2]"
    // We need to extract the elementary type of array from the information provided above
    // We want to make it like ["baseType"]["typeDescriptions"]
    nlohmann::json array_elementary_type =
      make_array_elementary_type(type_name);
    typet the_type;
    if(get_type_description(array_elementary_type, the_type))
      return true;

    assert(the_type.is_unsignedbv()); // assuming array size is unsigned bv
    std::string the_size = get_array_size(type_name);
    unsigned z_ext_value = std::stoul(the_size, nullptr);
    new_type = array_typet(
      the_type,
      constant_exprt(
        integer2binary(z_ext_value, bv_width(int_type())),
        integer2string(z_ext_value),
        int_type()));

    break;
  }
  case SolidityGrammar::TypeNameT::DynArrayTypeName:
  {
    // Deal with dynamic array
    exprt size_expr;
    if(type_name.contains("sizeExpr"))
    {
      const nlohmann::json &rtn_expr = type_name["sizeExpr"];
      // wrap it in an ImplicitCastExpr to convert LValue to RValue
      nlohmann::json implicit_cast_expr =
        make_implicit_cast_expr(rtn_expr, "LValueToRValue");
      if(get_expr(implicit_cast_expr, size_expr))
        return true;
    }
    else
    {
      new_type = empty_typet();
    }

    typet subtype;
    nlohmann::json array_elementary_type =
      make_array_elementary_type(type_name);
    if(get_type_description(array_elementary_type, subtype))
      return true;

    new_type = array_typet(subtype, size_expr);

    break;
  }
  default:
  {
    log_debug(
      "	@@@ got type name=SolidityGrammar::TypeNameT::{}",
      SolidityGrammar::type_name_to_str(type));
    assert(!"Unimplemented type in rule type-name");
    return true;
  }
  }

  // TODO: More var decl attributes checks:
  //    - Constant
  //    - Volatile
  //    - isRestrict

  return false;
}

bool solidity_convertert::get_func_decl_ref_type(
  const nlohmann::json &decl,
  typet &new_type)
{
  // For FunctionToPointer decay:
  // Get type when we make a function call:
  //  - FunnctionNoProto: x = nondet()
  //  - FunctionProto:    z = add(x, y)
  // Similar to the function get_type_description()
  SolidityGrammar::FunctionDeclRefT type =
    SolidityGrammar::get_func_decl_ref_t(decl);

  switch(type)
  {
  case SolidityGrammar::FunctionDeclRefT::FunctionNoProto:
  {
    code_typet type;

    // Return type
    const nlohmann::json &rtn_type = decl["returnParameters"];

    typet return_type;
    if(get_type_description(rtn_type, return_type))
      return true;

    type.return_type() = return_type;

    if(!type.arguments().size())
      type.make_ellipsis();

    new_type = type;
    break;
  }
  default:
  {
    log_debug("	@@@ Got type={}", SolidityGrammar::func_decl_ref_to_str(type));
    assert(!"Unimplemented type in auxiliary type to convert function call");
    return true;
  }
  }

  // TODO: More var decl attributes checks:
  //    - Constant
  //    - Volatile
  //    - isRestrict
  return false;
}

bool solidity_convertert::get_array_to_pointer_type(
  const nlohmann::json &type_descriptor,
  typet &new_type)
{
  // Function to get the base type in ArrayToPointer decay
  //  - unrolled the get_type...
  if(
    type_descriptor["typeString"].get<std::string>().find("uint8") !=
    std::string::npos)
  {
    new_type = unsigned_char_type();
    new_type.set("#cpp_type", "unsigned_char");
  }
  else
    assert(!"Unsupported types in ArrayToPinter decay");

  // TODO: More var decl attributes checks:
  //    - Constant
  //    - Volatile
  //    - isRestrict
  return false;
}

/**
 * @brief Populate the out `typet` parameter with the uint type specified by type parameter
 *
 * @param type The type of the uint to be poulated
 * @param out The variable that holds the resulting type
 * @return true iff population failed
 * @return false iff population was successful
 */
bool solidity_convertert::get_elementary_type_name_uint(
  SolidityGrammar::ElementaryTypeNameT &type,
  typet &out)
{
  const unsigned int uint_size = SolidityGrammar::uint_type_name_to_size(type);
  out = unsignedbv_typet(uint_size);

  return false;
}

/**
 * @brief Populate the out `typet` parameter with the int type specified by type parameter
 *
 * @param type The type of the int to be poulated
 * @param out The variable that holds the resulting type
 * @return false iff population was successful
 */
bool solidity_convertert::get_elementary_type_name_int(
  SolidityGrammar::ElementaryTypeNameT &type,
  typet &out)
{
  const unsigned int int_size = SolidityGrammar::int_type_name_to_size(type);
  out = signedbv_typet(int_size);

  return false;
}

bool solidity_convertert::get_elementary_type_name(
  const nlohmann::json &type_name,
  typet &new_type)
{
  // For Solidity rule elementary-type-name:
  // equivalent to clang's get_builtin_type()
  std::string c_type;
  SolidityGrammar::ElementaryTypeNameT type =
    SolidityGrammar::get_elementary_type_name_t(type_name);

  log_debug(
    "	@@@ got ElementaryType: SolidityGrammar::ElementaryTypeNameT::{}", type);

  switch(type)
  {
  // rule unsigned-integer-type
  case SolidityGrammar::ElementaryTypeNameT::UINT8:
  case SolidityGrammar::ElementaryTypeNameT::UINT16:
  case SolidityGrammar::ElementaryTypeNameT::UINT24:
  case SolidityGrammar::ElementaryTypeNameT::UINT32:
  case SolidityGrammar::ElementaryTypeNameT::UINT40:
  case SolidityGrammar::ElementaryTypeNameT::UINT48:
  case SolidityGrammar::ElementaryTypeNameT::UINT56:
  case SolidityGrammar::ElementaryTypeNameT::UINT64:
  case SolidityGrammar::ElementaryTypeNameT::UINT72:
  case SolidityGrammar::ElementaryTypeNameT::UINT80:
  case SolidityGrammar::ElementaryTypeNameT::UINT88:
  case SolidityGrammar::ElementaryTypeNameT::UINT96:
  case SolidityGrammar::ElementaryTypeNameT::UINT104:
  case SolidityGrammar::ElementaryTypeNameT::UINT112:
  case SolidityGrammar::ElementaryTypeNameT::UINT120:
  case SolidityGrammar::ElementaryTypeNameT::UINT128:
  case SolidityGrammar::ElementaryTypeNameT::UINT136:
  case SolidityGrammar::ElementaryTypeNameT::UINT144:
  case SolidityGrammar::ElementaryTypeNameT::UINT152:
  case SolidityGrammar::ElementaryTypeNameT::UINT160:
  case SolidityGrammar::ElementaryTypeNameT::UINT168:
  case SolidityGrammar::ElementaryTypeNameT::UINT176:
  case SolidityGrammar::ElementaryTypeNameT::UINT184:
  case SolidityGrammar::ElementaryTypeNameT::UINT192:
  case SolidityGrammar::ElementaryTypeNameT::UINT200:
  case SolidityGrammar::ElementaryTypeNameT::UINT208:
  case SolidityGrammar::ElementaryTypeNameT::UINT216:
  case SolidityGrammar::ElementaryTypeNameT::UINT224:
  case SolidityGrammar::ElementaryTypeNameT::UINT232:
  case SolidityGrammar::ElementaryTypeNameT::UINT240:
  case SolidityGrammar::ElementaryTypeNameT::UINT248:
  case SolidityGrammar::ElementaryTypeNameT::UINT256:
  {
    if(get_elementary_type_name_uint(type, new_type))
      return true;
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::INT8:
  case SolidityGrammar::ElementaryTypeNameT::INT16:
  case SolidityGrammar::ElementaryTypeNameT::INT24:
  case SolidityGrammar::ElementaryTypeNameT::INT32:
  case SolidityGrammar::ElementaryTypeNameT::INT40:
  case SolidityGrammar::ElementaryTypeNameT::INT48:
  case SolidityGrammar::ElementaryTypeNameT::INT56:
  case SolidityGrammar::ElementaryTypeNameT::INT64:
  case SolidityGrammar::ElementaryTypeNameT::INT72:
  case SolidityGrammar::ElementaryTypeNameT::INT80:
  case SolidityGrammar::ElementaryTypeNameT::INT88:
  case SolidityGrammar::ElementaryTypeNameT::INT96:
  case SolidityGrammar::ElementaryTypeNameT::INT104:
  case SolidityGrammar::ElementaryTypeNameT::INT112:
  case SolidityGrammar::ElementaryTypeNameT::INT120:
  case SolidityGrammar::ElementaryTypeNameT::INT128:
  case SolidityGrammar::ElementaryTypeNameT::INT136:
  case SolidityGrammar::ElementaryTypeNameT::INT144:
  case SolidityGrammar::ElementaryTypeNameT::INT152:
  case SolidityGrammar::ElementaryTypeNameT::INT160:
  case SolidityGrammar::ElementaryTypeNameT::INT168:
  case SolidityGrammar::ElementaryTypeNameT::INT176:
  case SolidityGrammar::ElementaryTypeNameT::INT184:
  case SolidityGrammar::ElementaryTypeNameT::INT192:
  case SolidityGrammar::ElementaryTypeNameT::INT200:
  case SolidityGrammar::ElementaryTypeNameT::INT208:
  case SolidityGrammar::ElementaryTypeNameT::INT216:
  case SolidityGrammar::ElementaryTypeNameT::INT224:
  case SolidityGrammar::ElementaryTypeNameT::INT232:
  case SolidityGrammar::ElementaryTypeNameT::INT240:
  case SolidityGrammar::ElementaryTypeNameT::INT248:
  case SolidityGrammar::ElementaryTypeNameT::INT256:
  {
    if(get_elementary_type_name_int(type, new_type))
      return true;
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::INT_LITERAL:
  {
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::BOOL:
  {
    new_type = bool_type();
    c_type = "bool";
    new_type.set("#cpp_type", c_type);
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::STRING:
  {
    size_t value_length = 128;

    new_type = array_typet(
      signed_char_type(),
      constant_exprt(
        integer2binary(value_length, bv_width(int_type())),
        integer2string(value_length),
        int_type()));
    break;
  }
  default:
  {
    log_debug(
      "	@@@ Got elementary-type-name={}",
      SolidityGrammar::elementary_type_name_to_str(type));
    assert(!"Unimplemented type in rule elementary-type-name");
    return true;
  }
  }

  return false;
}

bool solidity_convertert::get_parameter_list(
  const nlohmann::json &type_name,
  typet &new_type)
{
  // For Solidity rule parameter-list:
  //  - For non-empty param list, it may need to call get_elementary_type_name, since parameter-list is just a list of types
  std::string c_type;
  SolidityGrammar::ParameterListT type =
    SolidityGrammar::get_parameter_list_t(type_name);

  switch(type)
  {
  case SolidityGrammar::ParameterListT::EMPTY:
  {
    // equivalent to clang's "void"
    new_type = empty_typet();
    c_type = "void";
    new_type.set("#cpp_type", c_type);
    break;
  }
  case SolidityGrammar::ParameterListT::NONEMPTY:
  {
    assert(
      type_name["parameters"].size() ==
      1); // TODO: Fix me! assuming one return parameter
    const nlohmann::json &rtn_type =
      type_name["parameters"].at(0)["typeName"]["typeDescriptions"];
    return get_elementary_type_name(rtn_type, new_type);

    break;
  }
  default:
  {
    assert(!"Unimplemented type in rule parameter-list");
    return true;
  }
  }

  return false;
}

void solidity_convertert::get_state_var_decl_name(
  const nlohmann::json &ast_node,
  std::string &name,
  std::string &id)
{
  // Follow the way in clang:
  //  - For state variable name, just use the ast_node["name"]
  //  - For state variable id, add prefix "c:@"
  name =
    ast_node["name"]
      .get<
        std::
          string>(); // assume Solidity AST json object has "name" field, otherwise throws an exception in nlohmann::json
  id = "c:@" + name;
}

void solidity_convertert::get_var_decl_name(
  const nlohmann::json &ast_node,
  std::string &name,
  std::string &id)
{
  assert(
    current_functionDecl); // TODO: Fix me! assuming converting local variable inside a function
  // For non-state functions, we give it different id.
  // E.g. for local variable i in function nondet(), it's "c:overflow_2_nondet.c@55@F@nondet@i".
  name =
    ast_node["name"]
      .get<
        std::
          string>(); // assume Solidity AST json object has "name" field, otherwise throws an exception in nlohmann::json
  id = "c:" + get_modulename_from_path(absolute_path) + ".solast" +
       //"@"  + std::to_string(ast_node["scope"].get<int>()) +
       "@" + std::to_string(445) + "@" + "F" + "@" + current_functionName +
       "@" + name;
}

void solidity_convertert::get_function_definition_name(
  const nlohmann::json &ast_node,
  std::string &name,
  std::string &id)
{
  // Follow the way in clang:
  //  - For function name, just use the ast_node["name"]
  // assume Solidity AST json object has "name" field, otherwise throws an exception in nlohmann::json
  name = ast_node["name"].get<std::string>();
  id = name;
}

unsigned int solidity_convertert::add_offset(
  const std::string &src,
  unsigned int start_position)
{
  // extract the length from "start:length:index"
  std::string offset = src.substr(1, src.find(":"));
  // already added 1 in start_position
  unsigned int end_position = start_position + std::stoul(offset);
  return end_position;
}

std::string
solidity_convertert::get_src_from_json(const nlohmann::json &ast_node)
{
  // some nodes may have "src" inside a member json object
  // we need to deal with them case by case based on the node type
  SolidityGrammar::ExpressionT type =
    SolidityGrammar::get_expression_t(ast_node);
  switch(type)
  {
  case SolidityGrammar::ExpressionT::ImplicitCastExprClass:
  {
    assert(ast_node.contains("subExpr"));
    assert(ast_node["subExpr"].contains("src"));
    return ast_node["subExpr"]["src"].get<std::string>();
    break;
  }
  default:
  {
    assert(!"Unsupported node type when getting src from JSON");
    return "";
  }
  }
}

unsigned int solidity_convertert::get_line_number(
  const nlohmann::json &ast_node,
  bool final_position)
{
  // Solidity src means "start:length:index", where "start" represents the position of the first char byte of the identifier.
  std::string src = ast_node.contains("src")
                      ? ast_node["src"].get<std::string>()
                      : get_src_from_json(ast_node);

  std::string position = src.substr(0, src.find(":"));
  unsigned int byte_position = std::stoul(position) + 1;

  if(final_position)
    byte_position = add_offset(src, byte_position);

  // the line number can be calculated by counting the number of line breaks prior to the identifier.
  unsigned int loc = std::count(
                       contract_contents.begin(),
                       (contract_contents.begin() + byte_position),
                       '\n') +
                     1;
  return loc;
}

void solidity_convertert::get_location_from_decl(
  const nlohmann::json &ast_node,
  locationt &location)
{
  location.set_line(get_line_number(ast_node));
  location.set_file(
    absolute_path); // assume absolute_path is the name of the contrace file, since we ran solc in the same directory

  // To annotate local declaration within a function
  if(
    ast_node["nodeType"] == "VariableDeclaration" &&
    ast_node["stateVariable"] == false && ast_node["scope"] != global_scope_id)
  {
    assert(
      current_functionDecl); // must have a valid current function declaration
    location.set_function(
      current_functionName); // set the function where this local variable belongs to
  }
}

void solidity_convertert::get_start_location_from_stmt(
  const nlohmann::json &ast_node,
  locationt &location)
{
  std::string function_name;

  if(current_functionDecl)
    function_name = current_functionName;

  // The src manager of Solidity AST JSON is too encryptic.
  // For the time being we are setting it to "1".
  location.set_line(get_line_number(ast_node));
  location.set_file(
    absolute_path); // assume absolute_path is the name of the contrace file, since we ran solc in the same directory

  if(!function_name.empty())
    location.set_function(function_name);
}

void solidity_convertert::get_final_location_from_stmt(
  const nlohmann::json &ast_node,
  locationt &location)
{
  std::string function_name;

  if(current_functionDecl)
    function_name = current_functionName;

  // The src manager of Solidity AST JSON is too encryptic.
  // For the time being we are setting it to "1".
  location.set_line(get_line_number(ast_node, true));
  location.set_file(
    absolute_path); // assume absolute_path is the name of the contrace file, since we ran solc in the same directory

  if(!function_name.empty())
    location.set_function(function_name);
}

std::string solidity_convertert::get_modulename_from_path(std::string path)
{
  std::string filename = get_filename_from_path(path);

  if(filename.find_last_of('.') != std::string::npos)
    return filename.substr(0, filename.find_last_of('.'));

  return filename;
}

std::string solidity_convertert::get_filename_from_path(std::string path)
{
  if(path.find_last_of('/') != std::string::npos)
    return path.substr(path.find_last_of('/') + 1);

  return path; // for _x, it just returns "overflow_2.c" because the test program is in the same dir as esbmc binary
}

const nlohmann::json &solidity_convertert::find_decl_ref(int ref_decl_id)
{
  // First, search state variable nodes
  nlohmann::json &nodes = ast_json["nodes"];
  unsigned index = 0;
  for(nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end();
      ++itr, ++index)
  {
    if((*itr)["nodeType"] == "ContractDefinition") // contains AST nodes we need
      break;
  }

  nlohmann::json &ast_nodes = nodes.at(index)["nodes"];

  index = 0;
  for(nlohmann::json::iterator itrr = ast_nodes.begin();
      itrr != ast_nodes.end();
      ++itrr, ++index)
  {
    if((*itrr)["id"] == ref_decl_id)
      return ast_nodes.at(index);
  }

  // Then search "declarations" in current function scope
  const nlohmann::json &current_func = *current_functionDecl;
  if(current_func.contains("body"))
  {
    if(current_func["body"].contains("statements"))
    {
      // var declaration in local statements
      for(const auto &body_stmt : current_func["body"]["statements"].items())
      {
        const nlohmann::json &stmt = body_stmt.value();
        if(stmt["nodeType"] == "VariableDeclarationStatement")
        {
          for(const auto &local_decl : stmt["declarations"].items())
          {
            const nlohmann::json &the_decl = local_decl.value();
            if(the_decl["id"] == ref_decl_id)
              return the_decl;
          }
        }
      }
    }
    else
      assert(!"Unable to find the corresponding local variable decl. Function body  does not have statements.");
  }
  else
    assert(!"Unable to find the corresponding local variable decl. Current function does not have a function body.");

  // Search function parameter
  if(current_func.contains("parameters"))
  {
    if(current_func["parameters"]["parameters"].size())
    {
      // var decl in function parameter array
      for(const auto &param_decl :
          current_func["parameters"]["parameters"].items())
      {
        const nlohmann::json &param = param_decl.value();
        assert(param["nodeType"] == "VariableDeclaration");
        if(param["id"] == ref_decl_id)
          return param;
      }
    }
  }

  // if no matching state or local var decl, search decl in current_forStmt
  const nlohmann::json &current_for = *current_forStmt;
  if(current_for.contains("initializationExpression"))
  {
    if(current_for["initializationExpression"].contains("declarations"))
    {
      assert(current_for["initializationExpression"]["declarations"]
               .size()); // Assuming a non-empty declaration array
      const nlohmann::json &decls =
        current_for["initializationExpression"]["declarations"];
      for(const auto &init_decl : decls.items())
      {
        const nlohmann::json &the_decl = init_decl.value();
        if(the_decl["id"] == ref_decl_id)
          return the_decl;
      }
    }
    else
      assert(!"Unable to find the corresponding local variable decl. No local declarations found in current For-Statement");
  }
  else
    assert(!"Unable to find the corresponding local variable decl. Current For-Statement does not have any init.");

  assert(!"should not be here - no matching ref decl id found");
  return ast_json;
}

void solidity_convertert::get_default_symbol(
  symbolt &symbol,
  std::string module_name,
  typet type,
  std::string name,
  std::string id,
  locationt location)
{
  symbol.mode = mode;
  symbol.module = module_name;
  symbol.location = std::move(location);
  symbol.type = std::move(type);
  symbol.name = name;
  symbol.id = id;
}

symbolt *solidity_convertert::move_symbol_to_context(symbolt &symbol)
{
  return context.move_symbol_to_context(symbol);
}

void solidity_convertert::convert_expression_to_code(exprt &expr)
{
  if(expr.is_code())
    return;

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);

  expr.swap(code);
}

bool solidity_convertert::check_intrinsic_function(
  const nlohmann::json &ast_node)
{
  // function to detect special intrinsic functions, e.g. __ESBMC_assume
  return (
    ast_node["name"] == "__ESBMC_assume" ||
    ast_node["name"] == "__VERIFIER_assume");
}

nlohmann::json solidity_convertert::make_implicit_cast_expr(
  const nlohmann::json &sub_expr,
  std::string cast_type)
{
  // Since Solidity AST does not have type cast information about return values,
  // we need to manually make a JSON object and wrap the return expression in it.
  std::map<std::string, std::string> m = {
    {"nodeType", "ImplicitCastExprClass"},
    {"castType", cast_type},
    {"subExpr", {}}};
  nlohmann::json implicit_cast_expr = m;
  implicit_cast_expr["subExpr"] = sub_expr;

  return implicit_cast_expr;
}

nlohmann::json
solidity_convertert::make_pointee_type(const nlohmann::json &sub_expr)
{
  // Since Solidity function call node does not have enough information, we need to make a JSON object
  // manually create a JSON object to complete the conversions of function to pointer decay

  // make a mapping for JSON object creation latter
  // based on the usage of get_func_decl_ref_t() in get_func_decl_ref_type()
  nlohmann::json adjusted_expr;

  if(
    sub_expr["typeString"].get<std::string>().find("function") !=
    std::string::npos)
  {
    // Add more special functions here
    if(
      sub_expr["typeString"].get<std::string>().find("function ()") !=
        std::string::npos ||
      sub_expr["typeIdentifier"].get<std::string>().find(
        "t_function_assert_pure$") != std::string::npos ||
      sub_expr["typeIdentifier"].get<std::string>().find(
        "t_function_internal_pure$") != std::string::npos)
    {
      // e.g. FunctionNoProto: "typeString": "function () returns (uint8)" with () empty after keyword 'function'
      // "function ()" contains the function args in the parentheses.
      // make a type to behave like SolidityGrammar::FunctionDeclRefT::FunctionNoProto
      // Note that when calling "assert(.)", it's like "typeIdentifier": "t_function_assert_pure$......",
      //  it's also treated as "FunctionNoProto".
      auto j2 = R"(
        {
          "nodeType": "FunctionDefinition",
          "parameters":
            {
              "parameters" : []
            }
        }
      )"_json;
      adjusted_expr = j2;

      if(
        sub_expr["typeString"].get<std::string>().find("returns") !=
        std::string::npos)
      {
        // e.g. for typeString like:
        // "typeString": "function () returns (uint8)"
        // TODO: currently we only assume one parameters
        if(
          sub_expr["typeString"].get<std::string>().find("returns (uint8)") !=
          std::string::npos)
        {
          auto j2 = R"(
            {
              "typeIdentifier": "t_uint8",
              "typeString": "uint8"
            }
          )"_json;
          adjusted_expr["returnParameters"] = j2;
        }
        else
          assert(!"Unsupported return types in pointee");
      }
      else
      {
        // e.g. for typeString like:
        // "typeString": "function (bool) pure"
        auto j2 = R"(
          {
            "nodeType": "ParameterList",
            "parameters": []
          }
        )"_json;
        adjusted_expr["returnParameters"] = j2;
      }
    }
    else
      assert(!"Unsupported - detected function call with parameters");
  }
  else
    assert(!"Unsupported pointee - currently we only support the semantics of function to pointer decay");

  return adjusted_expr;
}

nlohmann::json solidity_convertert::make_callexpr_return_type(
  const nlohmann::json &type_descrpt)
{
  nlohmann::json adjusted_expr;
  if(
    type_descrpt["typeString"].get<std::string>().find("function") !=
    std::string::npos)
  {
    if(
      type_descrpt["typeString"].get<std::string>().find("returns") !=
      std::string::npos)
    {
      // e.g. for typeString like:
      // "typeString": "function () returns (uint8)"
      if(
        type_descrpt["typeString"].get<std::string>().find("returns (uint8)") !=
        std::string::npos)
      {
        auto j2 = R"(
            {
              "typeIdentifier": "t_uint8",
              "typeString": "uint8"
            }
          )"_json;
        adjusted_expr = j2;
      }
      else
        assert(!"Unsupported types in callee's return in CallExpr");
    }
    else
    {
      // Since Solidity allows multiple parameters and multiple returns for functions,
      // we need to use "parameters" in conjunction with "returnParameters" to convert.
      // the following configuration will lead to "void".
      auto j2 = R"(
          {
            "nodeType": "ParameterList",
            "parameters": []
          }
        )"_json;
      adjusted_expr = j2;
    }
  }
  else
    assert(!"Unsupported pointee - currently we only support the semantics of function to pointer decay");

  return adjusted_expr;
}

nlohmann::json solidity_convertert::make_array_elementary_type(
  const nlohmann::json &type_descrpt)
{
  // Function used to extract the elementary type of array
  // In order to keep the consistency and maximum the reuse of get_type_description function,
  // we used ["typeDescriptions"] instead of ["typeName"], despite the fact that the latter contains more information.
  // Although ["typeDescriptions"] also contains all the information needed, we have to do some
  // pre-processing, e.g. extract the
  nlohmann::json elementary_type;
  if(
    type_descrpt["typeString"].get<std::string>().find("uint8") !=
    std::string::npos)
  {
    auto j = R"(
      {
        "typeIdentifier": "t_uint8",
        "typeString": "uint8"
      }
    )"_json;
    elementary_type = j;
  }
  else
    assert(!"Unsupported array elementary type");

  return elementary_type;
}

nlohmann::json solidity_convertert::make_array_to_pointer_type(
  const nlohmann::json &type_descrpt)
{
  // Function to replace the content of ["typeIdentifier"] with "ArrayToPtr"
  // All the information in ["typeIdentifier"] should also be available in ["typeString"]
  std::string type_identifier = "ArrayToPtr";
  std::string type_string = type_descrpt["typeString"].get<std::string>();

  std::map<std::string, std::string> m = {
    {"typeIdentifier", type_identifier}, {"typeString", type_string}};
  nlohmann::json adjusted_type = m;

  return adjusted_type;
}

nlohmann::json solidity_convertert::add_dyn_array_size_expr(
  const nlohmann::json &type_descriptor,
  const nlohmann::json &dyn_array_node)
{
  nlohmann::json adjusted_descriptor;
  adjusted_descriptor = type_descriptor;
  // get the JSON object for size expr and merge it with the original type descriptor
  assert(dyn_array_node.contains("initialValue"));
  adjusted_descriptor.push_back(nlohmann::json::object_t::value_type(
    "sizeExpr", dyn_array_node["initialValue"]["arguments"][0]));
  return adjusted_descriptor;
}

std::string
solidity_convertert::get_array_size(const nlohmann::json &type_descrpt)
{
  const std::string s = type_descrpt["typeString"].get<std::string>();
  std::regex rgx(".*\\[([0-9]+)\\]");
  std::string the_size;

  std::smatch match;
  if(std::regex_search(s.begin(), s.end(), match, rgx))
  {
    std::ssub_match sub_match = match[1];
    the_size = sub_match.str();
  }
  else
    assert(!"Unsupported - Missing array size in type descriptor. Detected dynamic array?");

  return the_size;
}

bool solidity_convertert::is_dyn_array(const nlohmann::json &json_in)
{
  if(json_in.contains("typeIdentifier"))
  {
    if(
      json_in["typeIdentifier"].get<std::string>().find("dyn") !=
      std::string::npos)
    {
      return true;
    }
  }
  return false;
}
