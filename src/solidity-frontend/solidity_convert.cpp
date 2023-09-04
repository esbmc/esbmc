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
    current_functionName(""),
    current_contractName("")
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
  nlohmann::json &nodes = ast_json["nodes"];

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
      current_contractName = (*itr)["name"].get<std::string>();

      // set the ["Value"] for each member inside enum
      add_enum_member_val(*itr);

      // add a struct symbol for each contract
      // e.g. contract Base => struct Base
      if(get_struct_class(*itr))
        return true;

      if(convert_ast_nodes(*itr))
        return true; // 'true' indicates something goes wrong.

      // add implicit construcor function
      if(add_implicit_constructor())
        return true;

      // add function symbols to main
      if(move_functions_to_main(current_contractName))
        return true;
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
      "solidity",
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
  case SolidityGrammar::ContractBodyElementT::EnumDef:
  {
    break; // rule enum-definition
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
    "solidity",
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
    else if(
      init_value["isInlineArray"] != nullptr && init_value["isInlineArray"])
    {
      // TODO: make a function to convert inline array initialisation to index access assignment.
      int_literal_type =
        make_array_elementary_type(init_value["typeDescriptions"]);
    }
    else if(
      init_value["isInlineArray"] != nullptr && init_value["isInlineArray"])
    {
      // TODO: make a function to convert inline array initialisation to index access assignment.
      int_literal_type =
        make_array_elementary_type(init_value["typeDescriptions"]);
    }

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

bool solidity_convertert::get_struct_class(const nlohmann::json &contract_def)
{
  // Convert Contract => class => struct

  // 1. populate name, id
  std::string id, name;
  name = contract_def["name"];
  id = prefix + name;
  struct_typet t = struct_typet();
  t.tag(name);

  // 2. Check if the symbol is already added to the context, do nothing if it is
  // already in the context.
  if(context.find_symbol(id) != nullptr)
    return false;

  // 3. populate location
  locationt location_begin;
  get_location_from_decl(contract_def, location_begin);

  // 4. populate debug module name
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());
  current_fileName = debug_modulename;

  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, t, name, id, location_begin);

  symbol.is_type = true;
  symbolt &added_symbol = *move_symbol_to_context(symbol);

  // 5. populate fields(data member) and method(function)
  nlohmann::json ast_nodes = contract_def["nodes"];
  for(nlohmann::json::iterator itr = ast_nodes.begin(); itr != ast_nodes.end();
      ++itr)
  {
    SolidityGrammar::ContractBodyElementT type =
      SolidityGrammar::get_contract_body_element_t(*itr);

    switch(type)
    {
    case SolidityGrammar::ContractBodyElementT::StateVarDecl:
    {
      if(get_struct_class_fields(*itr, t))
        return true;
      break;
    }
    case SolidityGrammar::ContractBodyElementT::FunctionDef:
    {
      if(get_struct_class_method(*itr, t))
        return true;
      break;
    }
    case SolidityGrammar::ContractBodyElementT::EnumDef:
    {
      break;
    }
    default:
    {
      assert(!"Unimplemented type in rule contract-body-element");
      return true;
    }
    }
  }

  t.location() = location_begin;
  added_symbol.type = t;

  return false;
}

bool solidity_convertert::get_struct_class_fields(
  const nlohmann::json &ast_node,
  struct_typet &type)
{
  struct_typet::componentt comp;

  if(
    SolidityGrammar::get_access_t(ast_node) ==
    SolidityGrammar::VisibilityT::UnknownT)
    return false;

  if(get_var_decl_ref(ast_node, comp))
    return true;
  comp.type().set("#member_name", type.name());
  if(get_access_from_decl(ast_node, comp))
    return true;
  type.components().push_back(comp);

  return false;
}

bool solidity_convertert::get_struct_class_method(
  const nlohmann::json &ast_node,
  struct_typet &type)
{
  struct_typet::componentt comp;
  if(get_func_decl_ref(ast_node, comp))
    return false;

  if(comp.is_code() && to_code(comp).statement() == "skip")
    return false;

  type.methods().push_back(comp);
  return false;
}

void solidity_convertert::add_enum_member_val(nlohmann::json &contract_def)
{
  /*
  "nodeType": "EnumDefinition",
  "members": 
    [
      {
          "id": 2,
          "name": "SMALL",
          "nameLocation": "66:5:0",
          "nodeType": "EnumValue",
          "src": "66:5:0",
          "Value": 0 => new added object
      },
      {
          "id": 3,
          "name": "MEDIUM",
          "nameLocation": "73:6:0",
          "nodeType": "EnumValue",
          "src": "73:6:0",
          "Value": 1  => new added object
      },
    ] */

  nlohmann::json &ast_nodes = contract_def["nodes"];
  for(nlohmann::json::iterator itr = ast_nodes.begin(); itr != ast_nodes.end();
      ++itr)
  {
    if((*itr)["nodeType"] == "EnumDefinition")
    {
      int idx = 0;
      nlohmann::json &members = (*itr)["members"];
      for(nlohmann::json::iterator ittr = members.begin();
          ittr != members.end();
          ++ittr, ++idx)
      {
        (*ittr).push_back(
          nlohmann::json::object_t::value_type("Value", std::to_string(idx)));
      }
    }
  }
}

bool solidity_convertert::add_implicit_constructor()
{
  std::string name, id;
  name = id = current_contractName;

  if(context.find_symbol(id) != nullptr)
    return false;

  // an implicit constructor is an void empty function
  return get_default_function(name, id);
}

bool solidity_convertert::get_access_from_decl(
  const nlohmann::json &ast_node,
  struct_typet::componentt &comp)
{
  if(
    SolidityGrammar::get_access_t(ast_node) ==
    SolidityGrammar::VisibilityT::UnknownT)
    return true;

  std::string access = ast_node["visibility"].get<std::string>();
  comp.set_access(access);

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
  if(
    (*current_functionDecl)["name"].get<std::string>() == "" &&
    (*current_functionDecl)["kind"] == "constructor")
  {
    if(get_contract_name(
         (*current_functionDecl)["scope"], current_functionName))
      return true;
  }
  else
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
    log_debug("solidity", "  @@@ number of param decls: {}", num_param_decl);
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
    assert(!"Unimplemented - function parameter is array type");
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
    "solidity",
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
    log_debug("solidity", " \t@@@ CompoundStmt has {} statements", ctr);

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
    "solidity",
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
    log_debug("solidity", " \t@@@ DeclStmt group has {} decls", ctr);

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
    //  c). For multiple return type, the return statement represented as a tuple expression using a components field.
    //      Besides, tuple can only be declared literally. https://docs.soliditylang.org/en/latest/control-structures.html#assignment
    //      e.g. return (false, 123)
    assert(stmt.contains("expression"));

    typet return_type;
    assert(
      (*current_functionDecl)["returnParameters"]["id"].get<std::uint16_t>() ==
      stmt["functionReturnParameters"].get<std::uint16_t>());
    if(get_type_description(
         (*current_functionDecl)["returnParameters"], return_type))
      return true;

    nlohmann::json int_literal_type = nullptr;

    auto expr_type = SolidityGrammar::get_expression_t(stmt["expression"]);
    bool expr_is_literal = expr_type == SolidityGrammar::Literal;
    if(expr_is_literal)
      int_literal_type = make_return_type_from_typet(return_type);

    // 2. get return value
    code_returnt ret_expr;
    const nlohmann::json &rtn_expr = stmt["expression"];
    // wrap it in an ImplicitCastExpr to convert LValue to RValue
    nlohmann::json implicit_cast_expr =
      make_implicit_cast_expr(rtn_expr, "LValueToRValue");

    /* There could be case like
      {
      "expression": {
          "kind": "number",
          "nodeType": "Literal",
          "typeDescriptions": {
              "typeIdentifier": "t_rational_11_by_1",
              "typeString": "int_const 11"
          },
          "value": "12345"
      },
      "nodeType": "Return",
      }
      Therefore, we need to pass the int_literal_type value.
      */

    exprt val;
    if(get_expr(implicit_cast_expr, int_literal_type, val))
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
    current_forStmt = nullptr;
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
    "solidity",
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
  case SolidityGrammar::ExpressionT::ConditionalOperatorClass:
  {
    // for Ternary Operator (...?...:...) only
    if(get_conditional_operator_expr(expr, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::DeclRefExprClass:
  {
    if(expr["referencedDeclaration"] > 0)
    {
      // for Contract Type Identifier Only
      if(
        expr["typeDescriptions"]["typeString"].get<std::string>().find(
          "contract") != std::string::npos)
      {
        // TODO
        assert(!"we do not handle contract type identifier for now");
      }

      // Soldity uses +ve odd numbers to refer to var or functions declared in the contract
      assert(expr.contains("referencedDeclaration"));
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
        else if(decl["nodeType"] == "EnumValue")
        {
          if(get_enum_member_ref(decl, new_expr))
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
      "solidity",
      "	@@@ got Literal: SolidityGrammar::ElementaryTypeNameT::{}",
      SolidityGrammar::elementary_type_name_to_str(type_name));

    if(
      int_literal_type != nullptr &&
      int_literal_type["typeString"].get<std::string>().find("bytes") !=
        std::string::npos)
    {
      // int_literal_type["typeString"] could be
      //    "bytes1" ... "bytes32"
      //    "bytes storage ref"
      // e.g.
      //    bytes1 x = 0x12;
      //    bytes32 x = "string";
      //    bytes x = "string";
      //

      SolidityGrammar::ElementaryTypeNameT type =
        SolidityGrammar::get_elementary_type_name_t(int_literal_type);

      int byte_size;
      if(type == SolidityGrammar::ElementaryTypeNameT::BYTE_ARRAY)
        // dynamic bytes array, the type is set to uint_type()
        byte_size = 0;
      else
        byte_size = bytesn_type_name_to_size(type);

      // convert hex to decimal value and populate
      switch(type_name)
      {
      case SolidityGrammar::ElementaryTypeNameT::INT_LITERAL:
      {
        if(convert_hex_literal(the_value, new_expr, byte_size * 8))
          return true;
        break;
      }
      case SolidityGrammar::ElementaryTypeNameT::STRING_LITERAL:
      {
        std::string hex_val = expr["hexValue"].get<std::string>();

        // add padding
        for(int i = 0; i < byte_size; i++)
          hex_val += "00";
        hex_val.resize(byte_size * 2);

        if(convert_hex_literal(hex_val, new_expr, byte_size * 8))
          return true;
        break;
      }
      default:
        assert(!"Error occurred when handling bytes literal");
      }
      break;
    }

    switch(type_name)
    {
    case SolidityGrammar::ElementaryTypeNameT::INT_LITERAL:
    {
      assert(int_literal_type != nullptr);

      if(the_value.substr(0, 2) == "0x") // meaning hex-string
      {
        if(convert_hex_literal(the_value, new_expr))
          return true;
      }
      else if(convert_integer_literal(int_literal_type, the_value, new_expr))
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
    case SolidityGrammar::ElementaryTypeNameT::ADDRESS:
    case SolidityGrammar::ElementaryTypeNameT::ADDRESS_PAYABLE:
    {
      // 20 bytes
      if(convert_hex_literal(the_value, new_expr, 160))
        return true;
      break;
    }
    default:
      assert(!"Literal not implemented");
    }

    break;
  }
  case SolidityGrammar::ExpressionT::Tuple:
  {
    // This is an expr surrounded by parenthesis, we'll ignore it for
    // now, and check its subexpression
    for(const auto &arg : expr["components"].items())
    {
      if(get_expr(arg.value(), int_literal_type, new_expr))
        return true;
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
    log_debug("solidity", "  @@ num_args={}", num_args);

    // 4. Convert call arguments
    new_expr = call;

    break;
  }
  case SolidityGrammar::ExpressionT::ImplicitCastExprClass:
  {
    if(get_cast_expr(expr, new_expr, int_literal_type))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::IndexAccess:
  {
    // 1. get type, this is the base type of array
    typet t;
    if(get_type_description(expr["typeDescriptions"], t))
      return true;

    // for BYTESN, where the index access is read-only
    if(t.get("#sol_type").as_string().find("BYTES") != std::string::npos)
    {
      // e.g.
      //    bytes3 x = 0x123456
      //    bytes1 y = x[0]; // 0x12
      //    bytes1 z = x[1]; // 0x34
      // which equals to
      //    bytes1 z = bswap(x) >> 1 & 0xff
      // for bytes32 x = "test";
      //    x[10] == 0x00 due to the padding

      exprt src_val, src_offset, bswap, bexpr;

      const nlohmann::json &decl = find_decl_ref(
        expr["baseExpression"]["referencedDeclaration"].get<int>());
      if(get_var_decl_ref(decl, src_val))
        return true;

      if(get_expr(
           expr["indexExpression"], expr["typeDescriptions"], src_offset))
        return true;

      // extract particular byte based on idx (offset)
      bexpr = exprt("byte_extract_big_endian", src_val.type());
      bexpr.copy_to_operands(src_val, src_offset);

      solidity_gen_typecast(ns, bexpr, unsignedbv_typet(8));

      new_expr = bexpr;
      break;
    }

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
  case SolidityGrammar::ExpressionT::NewExpression:
  {
    // call the constructor
    if(get_constructor_call(expr, new_expr))
      return true;

    // break if the constructor call needs arguments
    // it would be reckoned as a memeber call.
    side_effect_expr_function_callt e =
      to_side_effect_expr_function_call(new_expr);
    if(e.arguments().size())
    {
      break;
    }

    side_effect_exprt tmp_obj("temporary_object", new_expr.type());
    codet code_expr("expression");
    code_expr.operands().push_back(new_expr);
    tmp_obj.initializer(code_expr);
    tmp_obj.location() = new_expr.location();
    new_expr.swap(tmp_obj);

    break;
  }
  case SolidityGrammar::ExpressionT::MemberCallClass:
  {
    assert(expr.contains("expression"));
    const nlohmann::json &callee_expr_json = expr["expression"];

    // Function symbol id is c:@C@referenced_function_contract_name@F@function_name#referenced_function_id
    // Using referencedDeclaration will point us to the original declared function. This works even for inherited function and overrided functions.

    const int caller_id =
      callee_expr_json["referencedDeclaration"].get<std::uint16_t>();
    const nlohmann::json caller_expr_json = find_decl_ref(caller_id);
    assert(caller_expr_json.contains("scope"));
    const int contract_id = caller_expr_json["scope"].get<std::uint16_t>();

    std::string ref_contract_name;
    get_contract_name(contract_id, ref_contract_name);

    std::string name, id;
    get_function_definition_name(caller_expr_json, name, id);

    if(context.find_symbol(id) == nullptr)
      // probably a built-in function
      // that is not supported yet
      return true;

    const symbolt s = *context.find_symbol(id);
    typet type = s.type;

    new_expr = exprt("symbol", type);
    new_expr.identifier(id);
    new_expr.cmt_lvalue(true);
    new_expr.name(name);
    new_expr.set("#member_name", prefix + ref_contract_name);

    // obtain the type of return value
    // It can be retrieved directly from the original function declaration
    typet t;
    if(get_type_description(caller_expr_json["returnParameters"], t))
      return true;

    side_effect_expr_function_callt call;
    call.function() = new_expr;
    call.type() = t;

    // populate params
    auto param_nodes = caller_expr_json["parameters"]["parameters"];
    unsigned num_args = 0;
    for(const auto &arg : expr["arguments"].items())
    {
      nlohmann::json param = nullptr;
      nlohmann::json::iterator itr = param_nodes.begin();
      if(itr != param_nodes.end())
      {
        if((*itr).contains("typeDescriptions"))
        {
          param = (*itr)["typeDescriptions"];
        }
        ++itr;
      }

      exprt single_arg;
      if(get_expr(arg.value(), param, single_arg))
        return true;

      call.arguments().push_back(single_arg);
      ++num_args;
    }
    log_debug("solidity", "  @@ num_args={}", num_args);

    new_expr = call;
    break;
  }
  case SolidityGrammar::ExpressionT::ElementaryTypeNameExpression:
  {
    // perform type conversion
    // e.g.
    // address payable public owner = payable(msg.sender);
    // or
    // uint32 a = 0x432178;
    // uint16 b = uint16(a); // b will be 0x2178 now

    assert(expr.contains("expression"));
    const nlohmann::json conv_expr = expr["expression"];
    typet type;
    exprt from_expr;

    // 1. get source expr
    // assume: only one argument
    if(get_expr(expr["arguments"][0], from_expr))
      return true;

    // 2. get target type
    if(get_type_description(conv_expr["typeDescriptions"], type))
      return true;

    // 3. generate the type casting expr
    convert_type_expr(ns, from_expr, type);

    new_expr = from_expr;
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

bool solidity_convertert::get_contract_name(
  const int ref_decl_id,
  std::string &contract_name)
{
  nlohmann::json &nodes = ast_json["nodes"];

  unsigned index = 0;
  for(nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end();
      ++itr, ++index)
  {
    if((*itr)["id"] == ref_decl_id)
    {
      contract_name = (*itr)["name"];
      return false;
    }
  }
  return true;
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
    "solidity",
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
  case SolidityGrammar::ExpressionT::BO_Mul:
  {
    if(t.is_floatbv())
      assert(!"Solidity does not support FP arithmetic as of v0.8.6.");
    else
      new_expr = exprt("*", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Div:
  {
    if(t.is_floatbv())
      assert(!"Solidity does not support FP arithmetic as of v0.8.6.");
    else
      new_expr = exprt("/", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Rem:
  {
    new_expr = exprt("mod", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Shl:
  {
    new_expr = exprt("shl", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Shr:
  {
    new_expr = exprt("shr", t);
    break;
  }
  case SolidityGrammar::BO_And:
  {
    new_expr = exprt("bitand", t);
    break;
  }
  case SolidityGrammar::BO_Xor:
  {
    new_expr = exprt("bitxor", t);
    break;
  }
  case SolidityGrammar::BO_Or:
  {
    new_expr = exprt("bitor", t);
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
  case SolidityGrammar::ExpressionT::BO_GE:
  {
    new_expr = exprt(">=", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_LE:
  {
    new_expr = exprt("<=", t);
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
  case SolidityGrammar::ExpressionT::BO_LAnd:
  {
    new_expr = exprt("and", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_LOr:
  {
    new_expr = exprt("or", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Pow:
  {
    // lhs**rhs => pow(lhs, rhs)
    // double pow(double base, double exponent)

    side_effect_expr_function_callt call_expr;

    exprt type_expr("symbol");
    type_expr.name("pow");
    type_expr.identifier("c:@F@pow");
    type_expr.location() = lhs.location();

    code_typet type;
    type.return_type() = double_type();
    type_expr.type() = type;

    call_expr.function() = type_expr;
    call_expr.type() = double_type();
    call_expr.set("#cpp_type", "double");

    solidity_gen_typecast(ns, lhs, double_type());
    solidity_gen_typecast(ns, rhs, double_type());
    call_expr.arguments().push_back(lhs);
    call_expr.arguments().push_back(rhs);

    new_expr = call_expr;

    return false;
  }
  default:
  {
    if(get_compound_assign_expr(expr, new_expr))
    {
      assert(!"Unimplemented binary operator");
      return true;
    }

    current_BinOp_type.pop();

    return false;
  }
  }

  // for bytes type
  if(
    lhs.type().get("#sol_type").as_string().find("BYTES") !=
      std::string::npos ||
    rhs.type().get("#sol_type").as_string().find("BYTES") != std::string::npos)
  {
    switch(opcode)
    {
    case SolidityGrammar::ExpressionT::BO_GT:
    case SolidityGrammar::ExpressionT::BO_LT:
    case SolidityGrammar::ExpressionT::BO_GE:
    case SolidityGrammar::ExpressionT::BO_LE:
    case SolidityGrammar::ExpressionT::BO_NE:
    case SolidityGrammar::ExpressionT::BO_EQ:
    {
      // e.g. cmp(0x74,  0x7500)
      // ->   cmp(0x74, 0x0075)
      exprt bwrhs, bwlhs;
      bwrhs = exprt("bswap", rhs.type());
      bwrhs.operands().push_back(rhs);
      rhs = bwrhs;

      bwlhs = exprt("bswap", lhs.type());
      bwlhs.operands().push_back(lhs);
      lhs = bwlhs;

      break;
    }
    case SolidityGrammar::ExpressionT::BO_Shl:
    {
      // e.g.
      //    bytes1 = 0x11
      //    x<<8 == 0x00
      new_expr.copy_to_operands(lhs, rhs);
      solidity_gen_typecast(ns, new_expr, lhs.type());

      return false;
    }
    default:
    {
      break;
    }
    }
  }

  // 4. Copy to operands
  new_expr.copy_to_operands(lhs, rhs);

  // Pop current_BinOp_type.push as we've finished this conversion
  current_BinOp_type.pop();

  return false;
}

bool solidity_convertert::get_compound_assign_expr(
  const nlohmann::json &expr,
  exprt &new_expr)
{
  // equivalent to clang_c_convertert::get_compound_assign_expr

  SolidityGrammar::ExpressionT opcode =
    SolidityGrammar::get_expr_operator_t(expr);

  switch(opcode)
  {
  case SolidityGrammar::ExpressionT::BO_AddAssign:
  {
    new_expr = side_effect_exprt("assign+");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_SubAssign:
  {
    new_expr = side_effect_exprt("assign-");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_MulAssign:
  {
    new_expr = side_effect_exprt("assign*");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_DivAssign:
  {
    new_expr = side_effect_exprt("assign_div");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_RemAssign:
  {
    new_expr = side_effect_exprt("assign_mod");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_ShlAssign:
  {
    new_expr = side_effect_exprt("assign_shl");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_ShrAssign:
  {
    new_expr = side_effect_exprt("assign_shr");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_AndAssign:
  {
    new_expr = side_effect_exprt("assign_bitand");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_XorAssign:
  {
    new_expr = side_effect_exprt("assign_bitxor");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_OrAssign:
  {
    new_expr = side_effect_exprt("assign_bitor");
    break;
  }
  default:
    return true;
  }

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

  assert(current_BinOp_type.size());
  const nlohmann::json &binop_type = *(current_BinOp_type.top());
  if(get_type_description(binop_type, new_expr.type()))
    return true;

  if(!lhs.type().is_pointer())
    solidity_gen_typecast(ns, rhs, lhs.type());

  new_expr.copy_to_operands(lhs, rhs);
  return false;
}

bool solidity_convertert::get_unary_operator_expr(
  const nlohmann::json &expr,
  const nlohmann::json &int_literal_type,
  exprt &new_expr)
{
  // TODO: Fix me! Currently just support prefix == true,e.g. pre-increment

  // 1. get UnaryOperation opcode
  SolidityGrammar::ExpressionT opcode =
    SolidityGrammar::get_unary_expr_operator_t(expr, expr["prefix"]);
  log_debug(
    "solidity",
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
  case SolidityGrammar::UO_PostDec:
  {
    new_expr = side_effect_exprt("postdecrement", uniop_type);
    break;
  }
  case SolidityGrammar::UO_PostInc:
  {
    new_expr = side_effect_exprt("postincrement", uniop_type);
    break;
  }
  case SolidityGrammar::ExpressionT::UO_Minus:
  {
    new_expr = exprt("unary-", uniop_type);
    break;
  }
  case SolidityGrammar::ExpressionT::UO_Not:
  {
    new_expr = exprt("bitnot", uniop_type);
    break;
  }

  case SolidityGrammar::ExpressionT::UO_LNot:
  {
    new_expr = exprt("not", bool_type());
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

bool solidity_convertert::get_conditional_operator_expr(
  const nlohmann::json &expr,
  exprt &new_expr)
{
  exprt cond;
  if(get_expr(expr["condition"], cond))
    return true;

  exprt then;
  if(get_expr(expr["trueExpression"], expr["typeDescriptions"], then))
    return true;

  exprt else_expr;
  if(get_expr(expr["falseExpression"], expr["typeDescriptions"], else_expr))
    return true;

  typet t;
  if(get_type_description(expr["typeDescriptions"], t))
    return true;

  exprt if_expr("if", t);
  if_expr.copy_to_operands(cond, then, else_expr);

  new_expr = if_expr;

  return false;
}

bool solidity_convertert::get_cast_expr(
  const nlohmann::json &cast_expr,
  exprt &new_expr,
  const nlohmann::json int_literal_type)
{
  // 1. convert subexpr
  exprt expr;
  if(get_expr(cast_expr["subExpr"], int_literal_type, expr))
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
  // TODO: Maybe can just type = expr.type() for other types as well. Need to make sure types are all set in get_expr (many functions are called multiple times to perform the same action).
  else
  {
    type = expr.type();
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

bool solidity_convertert::get_enum_member_ref(
  const nlohmann::json &decl,
  exprt &new_expr)
{
  assert(decl["nodeType"] == "EnumValue" && decl.contains("Value"));

  const std::string val = decl["Value"].get<std::string>();

  new_expr = constant_exprt(
    integer2binary(string2integer(val), bv_width(int_type())), val, int_type());

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
        std::string::npos ||
      type_name["typeString"].get<std::string>().find("contract") !=
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
  case SolidityGrammar::TypeNameT::ContractTypeName:
  {
    // e.g. ContractName tmp = new ContractName(Args);

    std::string constructor_name = type_name["typeString"].get<std::string>();
    size_t pos = constructor_name.find(" ");
    std::string id = prefix + constructor_name.substr(pos + 1);

    if(context.find_symbol(id) == nullptr)
      return true;

    const symbolt &s = *context.find_symbol(id);
    new_type = s.type;

    break;
  }
  case SolidityGrammar::TypeNameT::TypeConversionName:
  {
    // e.g.
    // uint32 a = 0x432178;
    // uint16 b = uint16(a); // b will be 0x2178 now
    //
    // "typeDescriptions": {
    //     "typeIdentifier": "t_type$_t_uint16_$",
    //     "typeString": "type(uint16)"
    // }

    nlohmann::json new_json;
    std::string typeIdentifier = type_name["typeIdentifier"].get<std::string>();
    std::string typeString = type_name["typeString"].get<std::string>();

    // convert it back to ElementaryTypeName by removing the "type" prefix
    std::size_t begin = typeIdentifier.find("$_");
    std::size_t end = typeIdentifier.rfind("_$");
    typeIdentifier = typeIdentifier.substr(begin + 2, end - begin - 2);

    begin = typeString.find("type(");
    end = typeString.rfind(")");
    typeString = typeString.substr(begin + 5, end - begin - 5);

    new_json["typeIdentifier"] = typeIdentifier;
    new_json["typeString"] = typeString;

    get_elementary_type_name(new_json, new_type);

    break;
  }
  case SolidityGrammar::TypeNameT::EnumTypeName:
  {
    new_type = enum_type();
    break;
  }
  default:
  {
    log_debug(
      "solidity",
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
    log_debug(
      "solidity",
      "	@@@ Got type={}",
      SolidityGrammar::func_decl_ref_to_str(type));
    //TODO: seem to be unnecessary, need investigate
    // assert(!"Unimplemented type in auxiliary type to convert function call");
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

bool solidity_convertert::get_elementary_type_name_bytesn(
  SolidityGrammar::ElementaryTypeNameT &type,
  typet &out)
{
  const unsigned int byte_num = SolidityGrammar::bytesn_type_name_to_size(type);
  out = signedbv_typet(byte_num * 8);

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
    "solidity",
    "	@@@ got ElementaryType: SolidityGrammar::ElementaryTypeNameT::{}",
    fmt::underlying(type));

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
    // for int_const type
    new_type = int_type();
    new_type.set("#cpp_type", "signed_char");
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
  case SolidityGrammar::ElementaryTypeNameT::ADDRESS:
  case SolidityGrammar::ElementaryTypeNameT::ADDRESS_PAYABLE:
  {
    //  An Address is a DataHexString of 20 bytes (uint160)
    // e.g. 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984
    // ops: <=, <, ==, !=, >= and >

    new_type = unsignedbv_typet(160);

    // for type conversion
    new_type.set("#sol_type", elementary_type_name_to_str(type));

    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::BYTES1:
  case SolidityGrammar::ElementaryTypeNameT::BYTES2:
  case SolidityGrammar::ElementaryTypeNameT::BYTES3:
  case SolidityGrammar::ElementaryTypeNameT::BYTES4:
  case SolidityGrammar::ElementaryTypeNameT::BYTES5:
  case SolidityGrammar::ElementaryTypeNameT::BYTES6:
  case SolidityGrammar::ElementaryTypeNameT::BYTES7:
  case SolidityGrammar::ElementaryTypeNameT::BYTES8:
  case SolidityGrammar::ElementaryTypeNameT::BYTES9:
  case SolidityGrammar::ElementaryTypeNameT::BYTES10:
  case SolidityGrammar::ElementaryTypeNameT::BYTES11:
  case SolidityGrammar::ElementaryTypeNameT::BYTES12:
  case SolidityGrammar::ElementaryTypeNameT::BYTES13:
  case SolidityGrammar::ElementaryTypeNameT::BYTES14:
  case SolidityGrammar::ElementaryTypeNameT::BYTES15:
  case SolidityGrammar::ElementaryTypeNameT::BYTES16:
  case SolidityGrammar::ElementaryTypeNameT::BYTES17:
  case SolidityGrammar::ElementaryTypeNameT::BYTES18:
  case SolidityGrammar::ElementaryTypeNameT::BYTES19:
  case SolidityGrammar::ElementaryTypeNameT::BYTES20:
  case SolidityGrammar::ElementaryTypeNameT::BYTES21:
  case SolidityGrammar::ElementaryTypeNameT::BYTES22:
  case SolidityGrammar::ElementaryTypeNameT::BYTES23:
  case SolidityGrammar::ElementaryTypeNameT::BYTES24:
  case SolidityGrammar::ElementaryTypeNameT::BYTES25:
  case SolidityGrammar::ElementaryTypeNameT::BYTES26:
  case SolidityGrammar::ElementaryTypeNameT::BYTES27:
  case SolidityGrammar::ElementaryTypeNameT::BYTES28:
  case SolidityGrammar::ElementaryTypeNameT::BYTES29:
  case SolidityGrammar::ElementaryTypeNameT::BYTES30:
  case SolidityGrammar::ElementaryTypeNameT::BYTES31:
  case SolidityGrammar::ElementaryTypeNameT::BYTES32:
  {
    if(get_elementary_type_name_bytesn(type, new_type))
      return true;

    // for type conversion
    new_type.set("#sol_type", elementary_type_name_to_str(type));
    new_type.set("#sol_bytes_size", bytesn_type_name_to_size(type));

    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::BYTE_ARRAY:
  {
    new_type = uint_type();
    new_type.set("#sol_type", elementary_type_name_to_str(type));

    break;
  }
  default:
  {
    log_debug(
      "solidity",
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
    return get_type_description(rtn_type, new_type);

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
  std::string contract_name;
  get_contract_name(ast_node["scope"], contract_name);

  if(ast_node["kind"].get<std::string>() == "constructor")
    name = contract_name;
  else
    name = ast_node["name"].get<std::string>();
  id = "c:@C@" + contract_name + "@F@" + name + "#" +
       i2string(ast_node["id"].get<std::int16_t>());
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
    if((*itr)["id"] == ref_decl_id)
      return find_constructor_ref(nodes.at(index)); // return construcor node

    if((*itr)["nodeType"] == "ContractDefinition") // contains AST nodes we need
    {
      nlohmann::json &ast_nodes = nodes.at(index)["nodes"];

      unsigned idx = 0;
      for(nlohmann::json::iterator itrr = ast_nodes.begin();
          itrr != ast_nodes.end();
          ++itrr, ++idx)
      {
        // for enum-member, as enum will not defined in the function
        if((*itrr)["nodeType"] == "EnumDefinition")
        {
          unsigned men_idx = 0;
          // enum cannot be empty
          nlohmann::json &mem_nodes = ast_nodes.at(idx)["members"];
          for(nlohmann::json::iterator mem_itr = mem_nodes.begin();
              mem_itr != mem_nodes.end();
              ++mem_itr, ++men_idx)
          {
            if((*mem_itr)["id"] == ref_decl_id)
              return mem_nodes.at(men_idx);
          }
        }
        if((*itrr)["id"] == ref_decl_id)
          return ast_nodes.at(idx);
      }
    }
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

const nlohmann::json &
solidity_convertert::find_constructor_ref(nlohmann::json &contract_def)
{
  nlohmann::json &nodes = contract_def["nodes"];
  unsigned index = 0;
  for(nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end();
      ++itr, ++index)
  {
    if((*itr)["kind"] == "constructor")
    {
      return nodes.at(index);
    }
  }
  // implicit constructor call
  return empty_json;
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
        adjusted_expr = R"(
            {
              "nodeType": "FunctionDefinition",
              "parameters":
                {
                  "parameters" : []
                }
            }
          )"_json;
        // e.g. for typeString like:
        // "typeString": "function () returns (uint8)"
        // use regex to capture the type and convert it to shorter form.
        std::smatch matches;
        std::regex e("returns \\((\\w+)\\)");
        std::string typeString = sub_expr["typeString"].get<std::string>();
        if(std::regex_search(typeString, matches, e))
        {
          auto j2 = nlohmann::json::parse(
            R"({
                "typeIdentifier": "t_)" +
            matches[1].str() + R"(",
                "typeString": ")" +
            matches[1].str() + R"("
              })");
          adjusted_expr["returnParameters"]["parameters"][0]
                       ["typeDescriptions"] = j2;
        }
        else if(
          sub_expr["typeString"].get<std::string>().find("returns (contract") !=
          std::string::npos)
        {
          // TODO: Fix me
          auto j2 = R"(
              {
                "nodeType": "ParameterList",
                "parameters": []
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

// Parse typet object into a typeDescriptions json
nlohmann::json solidity_convertert::make_return_type_from_typet(typet type)
{
  // Useful to get the width of a int literal type for return statement
  nlohmann::json adjusted_expr;
  if(type.is_signedbv() || type.is_unsignedbv())
  {
    std::string width = type.width().as_string();
    std::string type_name = (type.is_signedbv() ? "int" : "uint") + width;
    auto j2 = nlohmann::json::parse(
      R"({
              "typeIdentifier": "t_)" +
      type_name + R"(",
              "typeString": ")" +
      type_name + R"("
            })");
    adjusted_expr = j2;
  }
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
      // use regex to capture the type and convert it to shorter form.
      std::smatch matches;
      std::regex e("returns \\((\\w+)\\)");
      std::string typeString = type_descrpt["typeString"].get<std::string>();
      if(std::regex_search(typeString, matches, e))
      {
        auto j2 = nlohmann::json::parse(
          R"({
              "typeIdentifier": "t_)" +
          matches[1].str() + R"(",
              "typeString": ")" +
          matches[1].str() + R"("
            })");
        adjusted_expr = j2;
      }
      else if(
        type_descrpt["typeString"].get<std::string>().find(
          "returns (contract") != std::string::npos)
      {
        // TODO: Fix me. We treat contract as void
        auto j2 = R"(
              {
                "nodeType": "ParameterList",
                "parameters": []
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

bool solidity_convertert::get_constructor_call(
  const nlohmann::json &ast_node,
  exprt &new_expr)
{
  nlohmann::json callee_expr_json = ast_node["expression"];
  int ref_decl_id = callee_expr_json["typeName"]["referencedDeclaration"];
  exprt callee;

  const nlohmann::json constructor_ref = find_decl_ref(ref_decl_id);

  // Special handling of implicit constructor
  // since there is no ast nodes for implicit constructor
  if(constructor_ref.empty())
    return get_implicit_ctor_call(ref_decl_id, new_expr);

  if(get_func_decl_ref(constructor_ref, callee))
    return true;

  // obtain the type info
  std::string name, id;
  if(get_contract_name(ref_decl_id, name))
    return true;
  id = prefix + name;
  if(context.find_symbol(id) == nullptr)
    return true;

  const symbolt &s = *context.find_symbol(id);
  typet type = s.type;

  side_effect_expr_function_callt call;
  call.function() = callee;
  call.type() = type;

  auto param_nodes = constructor_ref["parameters"]["parameters"];
  unsigned num_args = 0;

  for(const auto &arg : ast_node["arguments"].items())
  {
    nlohmann::json param = nullptr;
    nlohmann::json::iterator itr = param_nodes.begin();
    if(itr != param_nodes.end())
    {
      if((*itr).contains("typeDescriptions"))
      {
        param = (*itr)["typeDescriptions"];
      }
      ++itr;
    }

    exprt single_arg;
    if(get_expr(arg.value(), param, single_arg))
      return true;

    call.arguments().push_back(single_arg);
    ++num_args;
  }

  // for adjustment
  call.set("constructor", 1);
  new_expr = call;

  return false;
}

bool solidity_convertert::get_implicit_ctor_call(
  const int ref_decl_id,
  exprt &new_expr)
{
  // to obtain the type info
  std::string name, id;
  if(get_contract_name(ref_decl_id, name))
    return true;
  id = name;
  if(context.find_symbol(id) == nullptr)
    return true;

  const symbolt &s = *context.find_symbol(id);
  typet type = s.type;

  new_expr = exprt("symbol", type);
  new_expr.identifier(id);
  new_expr.cmt_lvalue(true);
  new_expr.name(name);

  side_effect_expr_function_callt call;
  struct_typet tmp = struct_typet();
  call.function() = new_expr;
  call.type() = tmp;

  call.set("constructor", 1);
  new_expr = call;

  return false;
}

/*
  construct a void function with empty function body
  then add this function to symbol table
*/
bool solidity_convertert::get_default_function(
  const std::string name,
  const std::string id)
{
  nlohmann::json ast_node;
  auto j2 = R"(
              {
                "nodeType": "ParameterList",
                "parameters": []
              }
            )"_json;
  ast_node["returnParameters"] = j2;

  code_typet type;
  if(get_type_description(ast_node["returnParameters"], type.return_type()))
    return true;

  locationt location_begin;

  if(current_fileName == "")
    return true;
  std::string debug_modulename = current_fileName;

  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, type, name, id, location_begin);

  symbol.lvalue = true;
  symbol.is_extern = false;
  symbol.file_local = false;

  symbolt &added_symbol = *move_symbol_to_context(symbol);

  code_blockt body_exprt = code_blockt();
  added_symbol.value = body_exprt;

  type.make_ellipsis();
  added_symbol.type = type;

  return false;
}

void solidity_convertert::convert_type_expr(
  const namespacet &ns,
  exprt &src_expr,
  const typet &dest_type)
{
  if(
    src_expr.type().get("#sol_type").as_string().find("BYTES") !=
      std::string::npos &&
    dest_type.get("#sol_type").as_string().find("BYTES") != std::string::npos)
  {
    // 1. Fixed-size Bytes Converted to Smaller Types
    //    bytes2 a = 0x4326;
    //    bytes1 b = bytes1(a); // b will be 0x43
    // 2. Fixed-size Bytes Converted to Larger Types
    //    bytes2 a = 0x4235;
    //    bytes4 b = bytes4(a); // b will be 0x42350000
    // which equals to:
    //    new_type b = bswap(new_type)(bswap(x)))

    exprt bswap_expr, sub_bswap_expr;

    // 1. bswap
    sub_bswap_expr = exprt("bswap", src_expr.type());
    sub_bswap_expr.operands().push_back(src_expr);

    // 2. typecast
    solidity_gen_typecast(ns, sub_bswap_expr, dest_type);

    // 3. bswap back
    bswap_expr = exprt("bswap", sub_bswap_expr.type());
    bswap_expr.operands().push_back(sub_bswap_expr);

    src_expr = bswap_expr;
  }

  else
    solidity_gen_typecast(ns, src_expr, dest_type);
}

static inline void static_lifetime_init(const contextt &context, codet &dest)
{
  dest = code_blockt();

  // call designated "initialization" functions
  context.foreach_operand_in_order(
    [&dest](const symbolt &s)
    {
      if(s.type.initialization() && s.type.is_code())
      {
        code_function_callt function_call;
        function_call.function() = symbol_expr(s);
        dest.move_to_operands(function_call);
      }
    });
}

/*
  verify the contract as a whole
*/
bool solidity_convertert::move_functions_to_main(
  const std::string &contractName)
{
  // return if "function" is set or "contract" is unset
  if(
    !config.options.get_option("function").empty() ||
    config.options.get_option("contract").empty())
    return false;

  // return if it's not the target contract
  if(contractName != config.options.get_option("contract"))
    return false;

  codet init_code, body_code;
  static_lifetime_init(context, body_code);
  static_lifetime_init(context, init_code);

  //body_code.make_block();
  init_code.make_block();

  // get contract symbol "tag-contract"

  const std::string id = prefix + contractName;
  if(context.find_symbol(id) == nullptr)
    return true;

  const symbolt &contract = *context.find_symbol(id);

  // a contract should be a struct
  assert(contract.type.is_struct());

  // 1. call constructor()
  if(context.find_symbol(contractName) == nullptr)
    return true;
  const symbolt &constructor = *context.find_symbol(contractName);
  code_function_callt call;
  call.location() = constructor.location;
  call.function() = symbol_expr(constructor);
  init_code.move_to_operands(call);

  // 2. do while(nondet_uint())

  // 3. do if(nondet_uint()) then func()
  const struct_typet::componentst &methods =
    to_struct_type(contract.type).methods();
  for(const auto &method : methods)
  {
    // extract method(function)
    const std::string func_id = method.identifier().as_string();
    // skip constructor
    if(func_id == contractName)
      continue;

    if(context.find_symbol(func_id) == nullptr)
      return true;

    const symbolt &func = *context.find_symbol(func_id);

    // construct ifthenelse

    // guard: nondet_uint()
    exprt guard_expr("symbol");
    guard_expr.name("nondet_int");
    guard_expr.identifier("c:@F@nondet_int");
    guard_expr.location() = func.location;

    code_typet type;
    type.return_type() = int_type();
    guard_expr.type() = type;

    // then
    code_function_callt then_expr;
    then_expr.location() = func.location;
    then_expr.function() = symbol_expr(func);
    const code_typet::argumentst &arguments =
      to_code_type(func.type).arguments();
    then_expr.arguments().resize(
      arguments.size(), static_cast<const exprt &>(get_nil_irep()));

    // ifthenelse
    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(guard_expr, then_expr);

    // move to body code block
    body_code.operands().push_back(if_expr);
  }

  // cond
  exprt cond_expr("symbol");
  cond_expr.name("nondet_uint");
  cond_expr.identifier("c:@F@nondet_uint");
  cond_expr.location() = init_code.location();

  code_whilet code_while;
  code_while.cond() = cond_expr;
  code_while.body() = body_code;

  init_code.operands().push_back(code_while);

  // add "main"
  symbolt new_symbol;

  code_typet main_type;
  main_type.return_type() = empty_typet();
  std::string sol_id = "c:@" + contractName + "@F@sol_main";
  std::string sol_name = "sol_main";
  new_symbol.location = contract.location;
  std::string debug_modulename =
    get_modulename_from_path(contract.location.file().as_string());
  get_default_symbol(
    new_symbol,
    debug_modulename,
    main_type,
    sol_name,
    sol_id,
    contract.location);

  new_symbol.lvalue = true;
  new_symbol.is_extern = false;
  new_symbol.file_local = false;

  symbolt &added_symbol = *context.move_symbol_to_context(new_symbol);

  // no params
  main_type.make_ellipsis();
  added_symbol.type = main_type;
  added_symbol.value = init_code;

  // set "main" function
  config.main = "sol_main";

  return false;
}