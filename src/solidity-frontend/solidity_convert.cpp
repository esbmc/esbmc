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
#include <iomanip>

solidity_convertert::solidity_convertert(contextt &_context,
  nlohmann::json &_ast_json, const std::string &_sol_func, const messaget &msg):
  context(_context),
  ns(context),
  ast_json(_ast_json),
  sol_func(_sol_func),
  msg(msg),
  current_scope_var_num(1),
  current_functionDecl(nullptr),
  current_functionName("")
{
}

bool solidity_convertert::convert()
{
  // This function consists of two parts:
  //  1. First, we perform pattern-based verificaiton
  //  2. Then we populate the context with symbols annotated based on the each AST node, and hence prepare for the GOTO conversion.

  if (!ast_json.contains("nodes")) // check json file contains AST nodes as Solidity might change
    assert(!"JSON file does not contain any AST nodes");

  if (!ast_json.contains("absolutePath")) // check json file contains AST nodes as Solidity might change
    assert(!"JSON file does not contain absolutePath");

  absolute_path = ast_json["absolutePath"].get<std::string>();

  // By now the context should have the symbols of all ESBMC's intrinsics and the dummy main
  // We need to convert Solidity AST nodes to the equivalent symbols and add them to the context
  nlohmann::json nodes = ast_json["nodes"];

  bool found_contract_def = false;
  unsigned index = 0;
  nlohmann::json::iterator itr = nodes.begin();
  for (; itr != nodes.end(); ++itr, ++index)
  {
    // ignore the meta information and locate nodes in ContractDefinition
    std::string node_type = (*itr)["nodeType"].get<std::string>();
    if (node_type == "ContractDefinition") // contains AST nodes we need
    {
      found_contract_def = true;
      // pattern-based verification
      assert(itr->contains("nodes"));
      auto pattern_check = std::make_unique<pattern_checker>((*itr)["nodes"], sol_func, msg);
      if(pattern_check->do_pattern_check())
        return true; // 'true' indicates something goes wrong.
    }
  }
  assert(found_contract_def);

  // reasoning-based verification
  //assert(!"Continue with symbol annotations");
  index = 0;
  itr = nodes.begin();
  for (; itr != nodes.end(); ++itr, ++index)
  {
    std::string node_type = (*itr)["nodeType"].get<std::string>();
    if (node_type == "ContractDefinition") // rule source-unit
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
  nlohmann::json::iterator itr = ast_nodes.begin();
  for (; itr != ast_nodes.end(); ++itr, ++index)
  {
    nlohmann::json ast_node = *itr;
    std::string node_name = ast_node["name"].get<std::string>();
    std::string node_type = ast_node["nodeType"].get<std::string>();
    printf("@@ Converting node[%u]: name=%s, nodeType=%s ...\n",
        index, node_name.c_str(), node_type.c_str());
    print_json_array_element(ast_node, node_type, index);
    exprt dummy_decl;
    if(get_decl(ast_node, dummy_decl))
      return true;
  }

  assert(current_functionDecl == nullptr);

  return false;
}

bool solidity_convertert::get_decl(const nlohmann::json &ast_node, exprt &new_expr)
{
  new_expr = code_skipt();

  if (!ast_node.contains("nodeType"))
    assert(!"Missing \'nodeType\' filed in ast_node");

  SolidityGrammar::ContractBodyElementT type = SolidityGrammar::get_contract_body_element_t(ast_node);

  // based on each element as in Solidty grammar "rule contract-body-element"
  switch(type)
  {
    case SolidityGrammar::ContractBodyElementT::StateVarDecl:
    {
      return get_state_var_decl(ast_node, new_expr); // rule state-variable-declaration
    }
    case SolidityGrammar::ContractBodyElementT::FunctionDef:
    {
      return get_function_definition(ast_node, new_expr); // rule function-definition
    }
    default:
    {
      assert(!"Unimplemented type in rule contract-body-element");
      return true;
    }
  }

  assert(!"all AST node done?");

  return false;
}

bool solidity_convertert::get_state_var_decl(const nlohmann::json &ast_node, exprt &new_expr)
{
  // For Solidity rule state-variable-declaration:
  // 1. populate typet
  typet t;
  if (get_type_name(ast_node["typeName"], t)) // "type-name" as in state-variable-declaration
    return true;

  // 2. populate id and name
  std::string name, id;
  get_state_var_decl_name(ast_node, name, id);

  // 3. populate location
  locationt location_begin;
  get_location_from_decl(ast_node, location_begin);

  // 4. populate debug module name
  std::string debug_modulename = get_modulename_from_path(location_begin.file().as_string());

  // 5. set symbol attributes
  symbolt symbol;
  get_default_symbol(
    symbol,
    debug_modulename,
    t,
    name,
    id,
    location_begin);

  symbol.lvalue = true;
  symbol.static_lifetime = true; // TODO: hard coded for now, may need to change later
  symbol.is_extern = false;
  symbol.file_local = false;

  bool has_init = ast_node.contains("value");
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

  if (has_init)
  {
    assert(!"unimplemented - state var decl has init value");
  }

  decl.location() = location_begin;

  new_expr = decl;

  return false;
}

bool solidity_convertert::get_function_definition(const nlohmann::json &ast_node, exprt &new_expr)
{
  // For Solidity rule function-definition:
  // 1. Check fd.isImplicit() --- skipped since it's not applicable to Solidity
  // 2. Check fd.isDefined() and fd.isThisDeclarationADefinition()
  if (!ast_node["implemented"]) // TODO: for interface function, it's just a definition. Add something like "&& isInterface_JustDefinition()"
    return false;

  // 3. Set current_scope_var_num, current_functionDecl and old_functionDecl
  current_scope_var_num = 1;
  const nlohmann::json *old_functionDecl = current_functionDecl;
  current_functionDecl = &ast_node;
  current_functionName = ast_node["name"];

  // 4. Return type
  code_typet type;
  if (get_type_name(ast_node["returnParameters"], type.return_type()))
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

  // 8. populate "std::string debug_modulename"
  std::string debug_modulename = get_modulename_from_path(location_begin.file().as_string());

  // 9. Populate "symbol.static_lifetime", "symbol.is_extern" and "symbol.file_local"
  symbolt symbol;
  get_default_symbol(
    symbol,
    debug_modulename,
    type,
    name,
    id,
    location_begin);

  symbol.lvalue = true;
  symbol.is_extern = false; // TODO: hard coded for now, may need to change later
  symbol.file_local = false;

  // 10. Add symbol into the context
  symbolt &added_symbol = *move_symbol_to_context(symbol);

  // 11. Convert parameters, if no parameter, assume ellipis
  //  - Convert params before body as they may get referred by the statement in the body
  SolidityGrammar::ParameterListT params =
    SolidityGrammar::get_parameter_list_t(ast_node["parameters"]);
  if ( params == SolidityGrammar::ParameterListT::EMPTY )
    type.make_ellipsis();
  else
    assert(!"come back and continue - conversion of function arguments");

  added_symbol.type = type;

  // 12. Convert body and embed the body into the same symbol
  if (ast_node.contains("body"))
  {
    exprt body_exprt;
    get_block(ast_node["body"], body_exprt);

    added_symbol.value = body_exprt;
  }

  assert(!"done - finished all expr stmt in function?");

  // 13. Restore current_functionDecl
  current_functionDecl = old_functionDecl; // for __ESBMC_assume, old_functionDecl == null

  return false;
}

bool solidity_convertert::get_block(const nlohmann::json &block, exprt &new_expr)
{
  // For rule block
  static int call_block_times = 0; // TODO: remove debug
  locationt location;
  get_start_location_from_stmt(block, location);

  SolidityGrammar::BlockT type = SolidityGrammar::get_block_t(block);

  switch(type)
  {
    // equivalent to clang::Stmt::CompoundStmtClass
    // deal with a block of statements
    case SolidityGrammar::BlockT::Statement:
    {
      printf("	@@@ got Block: SolidityGrammar::BlockT::Statement, ");
      printf("  call_block_times=%d\n", call_block_times++);
      const nlohmann::json &stmts = block["statements"];

      code_blockt block;
      unsigned ctr = 0;
      // items() returns a key-value pair with key being the index
      for(auto const &stmt_kv : stmts.items())
      {
        exprt statement;
        //print_json_stmt_element(stmt_kv.value(), stmt_kv.value()["nodeType"], ctr);
        if(get_statement(stmt_kv.value(), statement))
          return true;

        //convert_expression_to_code(statement);
        //block.operands().push_back(statement);
        ++ctr;
      }
      printf(" \t @@@ CompoundStmt has %u statements\n", ctr);
      assert(!"done conversion all statements?");

      // TODO: Set the end location for blocks
#if 0
      locationt location_end;
      get_final_location_from_stmt(stmt, location_end);

      block.end_location(location_end);
#endif
      new_expr = block;
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

bool solidity_convertert::get_statement(const nlohmann::json &stmt, exprt &new_expr)
{
  // For rule statement
  // Since this is an additional layer of grammar rules compared to clang C, we do NOT set location here.
  // Just pass the new_expr reference to the next layer.
  static int call_stmt_times = 0; // TODO: remove debug

  SolidityGrammar::StatementT type = SolidityGrammar::get_statement_t(stmt);

  switch(type)
  {
    case SolidityGrammar::StatementT::ExpressionStatement:
    {
      printf("	@@@ got Stmt: SolidityGrammar::StatementT::ExpressionStatement, ");
      printf("  call_stmt_times=%d\n", call_stmt_times++);
      get_expr(stmt["expression"], new_expr);
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

bool solidity_convertert::get_expr(const nlohmann::json &expr, exprt &new_expr)
{
  // For rule expression
  // We need to do location settings to match clang C's number of times to set the locations when recurring
  static int call_expr_times = 0; // TODO: remove debug
  locationt location;
  get_start_location_from_stmt(expr, location);

  SolidityGrammar::ExpressionT type = SolidityGrammar::get_expression_t(expr);
  switch(type)
  {
    case SolidityGrammar::ExpressionT::BinaryOperatorClass:
    {
      printf("	@@@ got Expr: SolidityGrammar::ExpressionT::BinaryOperatorClass, ");
      printf("  call_expr_times=%d\n", call_expr_times++);
      if (get_binary_operator_expr(expr, new_expr))
        return true;

      break;
    }
    case SolidityGrammar::ExpressionT::DeclRefExprClass:
    {
      printf("	@@@ got Expr: SolidityGrammar::ExpressionT::DeclRefExprClass, ");
      printf("  call_expr_times=%d\n", call_expr_times++);
      const nlohmann::json &decl = find_decl_ref(expr["referencedDeclaration"]);
      //print_json(decl);

      if (get_decl_ref(decl, new_expr))
        return true;

      break;
    }
    case SolidityGrammar::ExpressionT::Literal:
    {
      printf("	@@@ got Expr: SolidityGrammar::ExpressionT::Literal, ");
      printf("  call_expr_times=%d\n", call_expr_times++);

      // TODO: Fix me! Assuming the context of BinaryOperator
      assert(current_BinOp_type.size());
      const nlohmann::json &binop_type = *(current_BinOp_type.top());
      assert(binop_type["typeString"] == "uint8");
      // make a type-name json for integer literal conversion
      std::string the_value = expr["value"].get<std::string>();
      std::map<std::string, std::string> m = {{"name", "uint8"},
                                              {"nodeType", "ElementaryTypeName"},
                                              {"value", the_value}};
      nlohmann::json integer_literal = m;

      if(convert_integer_literal(integer_literal, new_expr))
        return true;

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

bool solidity_convertert::get_binary_operator_expr(const nlohmann::json &expr, exprt &new_expr)
{
  // preliminary step:
  current_BinOp_type.push(&(expr["typeDescriptions"]));

  // 1. Convert LHS
  exprt lhs;
  if(get_expr(expr["leftHandSide"], lhs))
    return true;

  // 2. Convert RHS
  exprt rhs;
  if(get_expr(expr["rightHandSide"], rhs))
    return true;

  // 3. Get type
  typet t;

  // 4. Convert opcode

  // 5. Copy to operands

  // Pop current_BinOp_type.push as we've finished this conversion
  current_BinOp_type.pop();

  assert(!"done - BO?");
  //new_expr.copy_to_operands(lhs, rhs);
  return false;
}

bool solidity_convertert::get_decl_ref(const nlohmann::json &decl, exprt &new_expr)
{
  assert(decl["stateVariable"] == true); // assume referring to state variable. If not, use switch-case or if-else block.

  std::string name, id;
  get_state_var_decl_name(decl, name, id);

  typet type;
  if (get_type_name(decl["typeName"], type)) // "type-name" as in state-variable-declaration
    return true;

  new_expr = exprt("symbol", type);
  new_expr.identifier(id);
  new_expr.cmt_lvalue(true);
  new_expr.name(name);
  return false;
}

bool solidity_convertert::get_type_name(const nlohmann::json &type_name, typet &new_type)
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
      return get_parameter_list(type_name, new_type);
    }
    default:
    {
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

bool solidity_convertert::get_elementary_type_name(const nlohmann::json &type_name, typet &new_type)
{
  // For Solidity rule elementary-type-name:
  // equivalent to clang's get_builtin_type()
  std::string c_type;
  SolidityGrammar::ElementaryTypeNameT type = SolidityGrammar::get_elementary_type_name_t(type_name);

  switch(type)
  {
    // rule unsigned-integer-type
    case SolidityGrammar::ElementaryTypeNameT::UINT8:
    {
      new_type = unsigned_char_type();
      c_type = "unsigned_char";
      new_type.set("#cpp_type", c_type);
      break;
    }
    default:
    {
      assert(!"Unimplemented type in rule elementary-type-name");
      return true;
    }
  }

  return false;
}

bool solidity_convertert::get_parameter_list(const nlohmann::json &type_name, typet &new_type)
{
  // For Solidity rule parameter-list:
  //  - For non-empty param list, it may need to call get_elementary_type_name, since parameter-list is just a list of types
  std::string c_type;
  SolidityGrammar::ParameterListT type = SolidityGrammar::get_parameter_list_t(type_name);

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
      assert(!"come back and continue - Loop through non-empty param list and process them using get_elementary_type_name");
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
    std::string &name, std::string &id)
{
  // Follow the way in clang:
  //  - For state variable name, just use the ast_node["name"]
  //  - For state variable id, add prefix "c:@"
  name = ast_node["name"].get<std::string>(); // assume Solidity AST json object has "name" field, otherwise throws an exception in nlohmann::json
  id = "c:@" + name;
}

void solidity_convertert::get_function_definition_name(
    const nlohmann::json &ast_node,
    std::string &name, std::string &id)
{
  // Follow the way in clang:
  //  - For function name, just use the ast_node["name"]
  //  - For function id, add prefix "c:@F@"
  name = ast_node["name"].get<std::string>(); // assume Solidity AST json object has "name" field, otherwise throws an exception in nlohmann::json
  id = "c:@F@" + name;
}

void solidity_convertert::get_location_from_decl(const nlohmann::json &ast_node, locationt &location)
{
  // The src manager of Solidity AST JSON is too encryptic.
  // For the time being we are setting it to "1".
  location.set_line(1);
  location.set_file(absolute_path); // assume absolute_path is the name of the contrace file, since we ran solc in the same directory
}

void solidity_convertert::get_start_location_from_stmt(const nlohmann::json &stmt_node, locationt &location)
{
  std::string function_name;

  if (current_functionDecl)
    function_name = current_functionName;

  // The src manager of Solidity AST JSON is too encryptic.
  // For the time being we are setting it to "1".
  location.set_line(1);
  location.set_file(absolute_path); // assume absolute_path is the name of the contrace file, since we ran solc in the same directory

  if (!function_name.empty())
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

const nlohmann::json& solidity_convertert::find_decl_ref(int ref_decl_id)
{
  nlohmann::json& nodes = ast_json["nodes"];
  nlohmann::json::iterator itr = nodes.begin();
  unsigned index = 0;
  for (; itr != nodes.end(); ++itr, ++index)
  {
    if ( (*itr)["nodeType"] == "ContractDefinition" ) // contains AST nodes we need
      break;
  }

  nlohmann::json& ast_nodes = nodes.at(index)["nodes"];
  nlohmann::json::iterator itrr = ast_nodes.begin();
  index = 0;
  for (; itrr != ast_nodes.end(); ++itrr, ++index)
  {
    if ( (*itrr)["id"] == ref_decl_id)
    {
      //print_json(ast_nodes.at(index));
      return ast_nodes.at(index);
    }
  }

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
  symbol.mode = "C";
  symbol.module = module_name;
  symbol.location = std::move(location);
  symbol.type = std::move(type);
  symbol.name = name;
  symbol.id = id;
}

symbolt *solidity_convertert::move_symbol_to_context(symbolt &symbol)
{
  symbolt *s = context.find_symbol(symbol.id);
  if(s == nullptr)
  {
    if(context.move(symbol, s))
    {
      std::cerr << "Couldn't add symbol " << symbol.name
                << " to symbol table\n";
      symbol.dump();
      abort();
    }
  }
  else
  {
    // types that are code means functions
    if(s->type.is_code())
    {
      if(symbol.value.is_not_nil() && !s->value.is_not_nil())
        s->swap(symbol);
    }
    else if(s->is_type)
    {
      if(symbol.type.is_not_nil() && !s->type.is_not_nil())
        s->swap(symbol);
    }
  }

  return s;
}

// debug functions
void solidity_convertert::print_json(const nlohmann::json &json_in)
{
  printf("### JSON: ###\n");
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}

void solidity_convertert::print_json_element(const nlohmann::json &json_in, const unsigned index,
    const std::string &key, const std::string& json_name)
{
  printf("### %s element[%u] content: key=%s, size=%lu ###\n",
      json_name.c_str(), index, key.c_str(), json_in.size());
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}

void solidity_convertert::print_json_array_element(const nlohmann::json &json_in,
    const std::string& node_type, const unsigned index)
{
  printf("### node[%u]: nodeType=%s ###\n", index, node_type.c_str());
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}

void solidity_convertert::print_json_stmt_element(const nlohmann::json &json_in,
    const std::string& node_type, const unsigned index)
{
  printf("\t### stmt[%u]: nodeType=%s ###\n", index, node_type.c_str());
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}
